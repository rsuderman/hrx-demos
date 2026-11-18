#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Classic-Q4 witnesses from independent logical values and hand-encoded bytes.

The tiny direct-kernel checkpoint does not depend on model/dispatch plumbing.
The expanded route cases use the same canonical dyadic activation inputs as the
maintained matrix witnesses, so quantization recovers their values exactly.
"""

from pathlib import Path
import struct
import sys

from canonical_fixtures import bank, copy_body, f32, input_row


def weights(tag, width, columns):
    wire, logical = bytearray(), []
    for column in range(columns):
        row = []
        for block in range(width // 32):
            codes = [(j * 5 + block * 3 + column * 7) % 16 for j in range(32)]
            codes[0], codes[15], codes[16], codes[31] = 0, 7, 15, 1
            scale = (-1 if (column + block) % 2 else 1) * (column % 3 + 1) / 32
            minimum = ((column + block) % 5 - 2) / 8 if tag == 3 else 0.0
            # Real affine constant blocks: these fail if the correction is
            # multiplied by d, dropped, signed incorrectly or applied per block
            # instead of exactly once for each owned activation word.
            if tag == 3 and block % 4 == 1:
                scale, minimum = 0.0, -0.375 if column % 2 else 0.625
            wire.extend(struct.pack("<e", scale))
            if tag == 3:
                wire.extend(struct.pack("<e", minimum))
            wire.extend(codes[j] | (codes[j + 16] << 4) for j in range(16))
            row.extend(
                (q - 8) * scale if tag == 2 else q * scale + minimum for q in codes
            )
        logical.append(row)
    return wire, logical


def packed_activations(values, width):
    wire = bytearray()
    for row in values:
        for group in range(width // 128):
            metadata, payload = bytearray(), bytearray()
            for inner in range(4):
                block = row[group * 128 + inner * 32 : group * 128 + (inner + 1) * 32]
                scale = max(abs(x) for x in block) / 127
                codes = [round(x / scale) for x in block]
                assert all(code * scale == x for code, x in zip(codes, block))
                metadata.extend(struct.pack("<ee", scale, sum(codes) * scale))
                payload.extend(code & 255 for code in codes)
            wire.extend(metadata + payload)
    return wire


def emit(destination, tiny=False):
    out = ["// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n"]
    for name in ("q4_0", "q4_1"):
        out.append(
            f"config.def @ggml.linear_{name}_q8_1_x4.token_capacity = 512 : index\n"
        )
        out.append(
            f"config.def @ggml.linear_{name}_q8_1_x4.output_capacity = 262144 : index\n"
        )
        out.append(
            f"kernel.decl @ggml_linear_{name}_q8_1_x4(%T:index,%K:index,%N:index) launch(%T:index,%K:index,%N:index,%input:buffer,%weight:buffer,%output:buffer)\n"
        )
    if not tiny:
        for name in (
            "q4k",
            "q5_0",
            "q5k",
            "q6k",
            "q8_0",
            "iq4_xs",
            "iq4_nl",
            "iq3_s",
            "q3k",
        ):
            out.append(
                f"config.def @ggml.linear_{name}_q8_1_x4.token_capacity = 512 : index\n"
            )
            out.append(
                f"config.def @ggml.linear_{name}_q8_1_x4.output_capacity = 262144 : index\n"
            )
        out.append(
            "config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n"
        )
        out.append(
            "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%B:index,%count:buffer,%input:buffer,%weight:view<[%B]xi8>,%output:buffer)\n"
        )
    cases = [("direct", tag, 1, 1, 128, 3) for tag in (2, 3)]
    if not tiny:
        cases += [
            ("direct", tag, 2, 2, width, 13) for tag in (2, 3) for width in (256, 9728)
        ]
        cases += [
            ("matrix", tag, rows, active, 256, 13)
            for tag in (2, 3)
            for rows, active in (
                (1, 1),
                (8, 0),
                (8, 1),
                (8, 8),
                (16, 16),
                (17, 0),
                (17, 1),
                (17, 17),
            )
        ]
    for mode, tag, rows, active, width, columns in cases:
        name = f"classic_{mode}_q{tag}_t{rows}_a{active}_k{width}_n{columns}"
        wire, logical = weights(tag, width, columns)
        raw = [input_row(row, width) for row in range(rows)]
        anchors = [
            f32(sum(x * w for x, w in zip(raw[row], channel))) if row < active else 0.0
            for row in range(rows)
            for channel in logical
        ]
        input_bytes = (
            packed_activations(raw, width)
            if mode == "direct"
            else b"".join(struct.pack("<f", x) for row in raw for x in row)
        )
        output_elements = rows * columns
        wb, ab = len(wire), len(input_bytes)
        out.extend(
            (
                bank(name + ".input", input_bytes),
                bank(name + ".weight", wire),
                bank(name + ".anchor", b"".join(struct.pack("<f", x) for x in anchors)),
            )
        )
        out.append(
            f"func.def @{name}.init(%input:buffer,%weight:buffer) {{\n"
            + copy_body(name + ".input", "input", ab, "i8")
            + copy_body(name + ".weight", "weight", wb, "i8")
            + "  func.return\n}\n"
        )
        out.append(
            f"func.def @{name}.expected(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{\n"
            + copy_body(name + ".anchor", "output", output_elements, "f32")
            + "  func.return\n}\n"
        )
        if mode == "direct":
            invoke = f"kernel.launch @ggml_linear_q4_{tag-2}_q8_1_x4[%T,%K,%N](%T,%K,%N,%input,%weight,%output) : [index,index,index](index,index,index,buffer,buffer,buffer)"
        else:
            invoke = f"""%tag = index.constant {tag} : index
  %B = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%B]xi8>
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%B,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%B]xi8>,buffer)"""
        out.append(
            f"""command.def @{name}.actual() launch(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{
  %T = index.constant {rows} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  {invoke}
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %input_owner = check.generate.fill value(-27) : tensor<{ab+512}xi8>
    %input = check.tensor.view %input_owner offset(256) : tensor<{ab+512}xi8> -> tensor<{ab}xi8>
    %weight_owner = check.generate.fill value(-35) : tensor<{wb+512}xi8>
    %weight = check.tensor.view %weight_owner offset(256) : tensor<{wb+512}xi8> -> tensor<{wb}xi8>
    check.generate<@{name}.init>(%input,%weight) : (tensor<{ab}xi8>,tensor<{wb}xi8>) -> ()
    %owner = check.generate.fill value(-99.0) : tensor<{output_elements+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{output_elements+8}xf32> -> tensor<{output_elements}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{output_elements+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({(output_elements+4)*4}) : tensor<{output_elements+8}xf32> -> tensor<4xf32>
    check.compare<@{name}.actual,@{name}.expected>(%count,%input,%weight,%output) : (tensor<3xi32>,tensor<{ab}xi8>,tensor<{wb}xi8>,tensor<{output_elements}xf32>) -> () {{
      check.expect.bitwise actual(%owner) expected(%owner) : tensor<{output_elements+8}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{ab+512}xi8>
      check.expect.bitwise actual(%weight_owner) expected(%weight_owner) : tensor<{wb+512}xi8>
    }}
  }}
  check.return
}}
"""
        )
    Path(destination).write_text("".join(out))


if __name__ == "__main__":
    emit(sys.argv[1], tiny="--tiny" in sys.argv[2:])
