#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Independent signed Q3_K encodings and canonical-Q8 projection witnesses.

Expected weights start as signed codes/scales, not a production decoder result.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

SCALES = (-32, -31, -17, -16, -1, 0, 1, 15, 16, 17, 23, 24, 28, 29, 30, 31)


def helpers(path):
    spec = importlib.util.spec_from_file_location("canonical_fixture_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(h, row, width):
    factor = (row % 3 + 1) * (-1 if row % 2 else 1)
    recipe = (
        127 / 128,
        321 / 1024,
        -28 / 128,
        -56 / 128,
        53 / 128,
        -173 / 1024,
        -11 / 128,
        31 / 128,
    )
    return [h.f32(factor * recipe[j % 8]) for j in range(width)]


def canonical(h, source):
    result = []
    for begin in range(0, len(source), 32):
        block = source[begin : begin + 32]
        d = h.f32(max(map(abs, block)) / 127)
        inverse = h.f32(1 / d) if d else 0.0
        scaled = [h.f32(value * inverse) for value in block]
        codes = [math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5) for x in scaled]
        assert all(-127 <= x <= 127 for x in codes)
        result.extend(h.f32(code * h.f16(d)) for code in codes)
    return result


def weights(h, width, columns, fault=None):
    wire, logical = bytearray(), []
    for channel in range(columns):
        row = []
        for block in range(width // 256):
            d = (-1 if (channel + block) % 2 else 1) * (channel % 3 + 1) / 65536
            assert h.f16(d) == d
            scales = [SCALES[(s + channel + block) % 16] for s in range(16)]
            hmask, qs, packed_scales = bytearray(32), bytearray(64), bytearray(12)
            for s, scale in enumerate(scales):
                biased = scale + 32
                packed_scales[s % 8] |= (biased & 15) << (4 * (s // 8))
                packed_scales[8 + s % 4] |= (biased >> 4) << (2 * (s // 4))
            for j in range(256):
                group, lane = divmod(j, 32)
                code = (lane * 3 + group * 5 + channel + block) % 8 - 4
                biased = code + 4
                qs[32 * (group // 4) + lane] |= (biased & 3) << (2 * (group % 4))
                hmask[lane] |= (biased >> 2) << group
                scale = scales[j // 16]
                if fault == "hmask_polarity":
                    code += -4 if code >= 0 else 4
                elif fault == "ignore_scale_high":
                    scale = (scale + 32) % 16 - 32
                elif fault == "swap_low_planes":
                    code = ((biased & 1) << 1 | (biased & 2) >> 1 | biased & 4) - 4
                row.append(h.f32(h.f32(d * scale) * code))
            wire.extend(hmask + qs + packed_scales + struct.pack("<e", d))
        logical.append(row)
    assert len(wire) == columns * (width // 256) * 110
    return wire, logical


def audit(h):
    _, logical = weights(h, 512, 7)
    source = inputs(h, 0, 512)
    rounded = canonical(h, source)
    records = []
    for fault in (
        "raw_input",
        "hmask_polarity",
        "ignore_scale_high",
        "swap_low_planes",
    ):
        _, wrong = weights(h, 512, 7, fault=fault)
        wrong_input = source if fault == "raw_input" else rounded
        samples = []
        for channel, values in enumerate(logical):
            expected = h.f32(math.fsum(x * w for x, w in zip(rounded, values)))
            mutant = h.f32(
                math.fsum(x * w for x, w in zip(wrong_input, wrong[channel]))
            )
            budget = 0.00001 + 0.00001 * abs(expected)
            samples.append(
                dict(
                    channel=channel,
                    expected=expected,
                    mutant=mutant,
                    separation=abs(expected - mutant) / budget,
                )
            )
        assert max(row["separation"] for row in samples) > 4, fault
        records.append(dict(fault=fault, samples=samples))
    return records


def emit(h, mode, destination):
    if mode == "audit":
        Path(destination).write_text(json.dumps(audit(h), indent=2) + "\n")
        return
    assert mode in ("tiny", "full")
    cases = [(1, 1, 256, 7)]
    if mode == "full":
        cases = [
            (1, 1, 256, 7),
            (1, 1, 512, 65),
            (1, 1, 5120, 7),
            (1, 1, 17408, 7),
            (8, 0, 256, 1),
            (8, 1, 512, 7),
            (8, 8, 512, 7),
            (16, 16, 512, 7),
            (17, 0, 256, 1),
            (17, 1, 512, 7),
            (17, 17, 512, 7),
            (65, 17, 256, 7),
        ]
    out = ["// Copyright 2026 The HRX Authors\n// SPDX-License-Identifier: MIT\n"]
    out.append(
        "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%weight:view<[%bytes]xi8>,%output:buffer)\n"
    )
    for tag in (
        "q4k",
        "q5_0",
        "q5k",
        "q6k",
        "q8_0",
        "iq4_xs",
        "q4_0",
        "q4_1",
        "q3k",
        "iq4_nl",
        "iq3_s",
    ):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    for rows, active, width, columns in cases:
        name = f"q3k_t{rows}_a{active}_k{width}_n{columns}"
        padded = rows if rows <= 16 else (rows + 63) // 64 * 64
        encoded, logical = weights(h, width, columns)
        original = [
            inputs(h, r, width) if r < active else [math.nan] * width
            for r in range(rows)
        ]
        rounded = [
            canonical(h, row) if r < active else [] for r, row in enumerate(original)
        ]
        expected = [
            (
                h.f32(math.fsum(x * y for x, y in zip(rounded[r], w)))
                if r < active
                else 0.0
            )
            for r in range(rows)
            for w in logical
        ]
        wb = len(encoded)
        out.extend(
            (
                h.bank(
                    name + ".input",
                    b"".join(struct.pack("<f", x) for row in original for x in row),
                ),
                h.bank(name + ".weight", encoded),
                h.bank(
                    name + ".expected_data",
                    b"".join(struct.pack("<f", x) for x in expected),
                ),
            )
        )
        out.append(
            f"func.def @{name}.init(%input:buffer,%weight:buffer) {{\n"
            + h.copy_body(name + ".input", "input", rows * width, "f32")
            + h.copy_body(name + ".weight", "weight", wb, "i8")
            + "  func.return\n}\n"
        )
        out.append(
            f"func.def @{name}.expected(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{\n"
            + h.copy_body(name + ".expected_data", "output", rows * columns, "f32")
            + "  func.return\n}\n"
        )
        tail_views, tail_checks = "", ""
        for label, start, size in (
            ("inactive", active * columns, (rows - active) * columns),
            ("padding", rows * columns, (padded - rows) * columns),
        ):
            if size:
                tail_views += f"    %{label} = check.tensor.view %output offset({start*4}) : tensor<{padded*columns}xf32> -> tensor<{size}xf32>\n"
                tail_checks += f"      check.expect.bitwise actual(%{label}) expected(%{label}) : tensor<{size}xf32>\n"
        out.append(
            f"""command.def @{name}.actual() launch(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{
  %T = index.constant {rows} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %tag = index.constant 11 : index
  %bytes = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %count_owner = check.generate.fill value({active}) : tensor<11xi32>
    %count = check.tensor.view %count_owner offset(16) : tensor<11xi32> -> tensor<3xi32>
    %input_owner = check.generate.fill value(-777.0) : tensor<{rows*width+8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{rows*width+8}xf32> -> tensor<{rows*width}xf32>
    %weight_owner = check.generate.fill value(-23) : tensor<{wb+512}xi8>
    %weight = check.tensor.view %weight_owner offset(256) : tensor<{wb+512}xi8> -> tensor<{wb}xi8>
    check.generate<@{name}.init>(%input,%weight) : (tensor<{rows*width}xf32>,tensor<{wb}xi8>) -> ()
    %owner = check.generate.fill value(-99.0) : tensor<{padded*columns+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{padded*columns+8}xf32> -> tensor<{padded*columns}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({(padded*columns+4)*4}) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>
{tail_views}    check.compare<@{name}.actual,@{name}.expected>(%count,%input,%weight,%output) : (tensor<3xi32>,tensor<{rows*width}xf32>,tensor<{wb}xi8>,tensor<{padded*columns}xf32>) -> () {{
      check.expect.approximate actual(%owner) expected(%owner) atol(0.00001) rtol(0.00001) nan(same) : tensor<{padded*columns+8}xf32>,tensor<{padded*columns+8}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>
      check.expect.bitwise actual(%count_owner) expected(%count_owner) : tensor<11xi32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{rows*width+8}xf32>
      check.expect.bitwise actual(%weight_owner) expected(%weight_owner) : tensor<{wb+512}xi8>
{tail_checks}    }}
  }}
  check.return
}}
"""
        )
    Path(destination).write_text("".join(out))


if __name__ == "__main__":
    helper_path, mode, destination = sys.argv[1:]
    emit(helpers(helper_path), mode, destination)
