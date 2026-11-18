#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""IQ3_S logical-grid/sign/odd-scale anchors for the canonical shared matrix.

The small named codebook anchors are independent literal examples of the GGML
four-magnitude grid, not values obtained from the production decoder or table.
Only wire encoding and the owned canonical activation precision are modeled.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

# Cover both table endpoints and the low8/ninth-bit boundary. Adjacent entries
# 255/256/257 ensure an ignored high index bit cannot reproduce these anchors.
GRID = (
    (0, (1, 1, 1, 1)),
    (1, (3, 1, 1, 1)),
    (2, (5, 1, 1, 1)),
    (3, (11, 1, 1, 1)),
    (4, (15, 1, 1, 1)),
    (5, (1, 3, 1, 1)),
    (6, (3, 3, 1, 1)),
    (7, (5, 3, 1, 1)),
    (255, (1, 5, 9, 5)),
    (256, (7, 5, 9, 5)),
    (257, (5, 7, 9, 5)),
    (511, (1, 1, 15, 15)),
)


def helpers(path):
    spec = importlib.util.spec_from_file_location("canonical_fixture_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(h, row, width):
    factor = (row % 3 + 1) * (-1 if row % 2 else 1)
    # 321/1024 distinguishes the required canonical Q8 activation packet from
    # substituting the donor's original raw F32 input.
    pattern = (127 / 128, 321 / 1024, -28 / 128, -56 / 128)
    return [h.f32(factor * pattern[j % 4]) for j in range(width)]


def canonical(h, values):
    result = []
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        d = h.f32(max(map(abs, block)) / 127)
        inv = h.f32(1 / d) if d else 0.0
        scaled = [h.f32(value * inv) for value in block]
        codes = [math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5) for x in scaled]
        scale = h.f16(d)
        result.extend(h.f32(code * scale) for code in codes)
    return result


def weights(h, width, columns, fault=None):
    wire, logical = bytearray(), []
    for column in range(columns):
        row = []
        for block in range(width // 256):
            scale = (-1 if (column + block) % 2 else 1) * (column % 3 + 1) / 4096
            if column % 7 == 6:
                scale = 0.0
            payload = bytearray(110)
            struct.pack_into("<e", payload, 0, scale)
            for group in range(8):
                nibble = (group * 5 + column + block * 3) % 16
                payload[106 + group // 2] |= nibble << (4 * (group % 2))
                odd = 2 * nibble + 1
                if fault == "even_scale":
                    odd = 2 * nibble
                group_values = []
                signs = 0
                for word in range(8):
                    number, magnitudes = GRID[
                        (word + group * 3 + block + column) % len(GRID)
                    ]
                    payload[2 + group * 8 + word] = number & 255
                    payload[66 + group] |= (number >> 8) << word
                    if fault == "ignore_ninth_bit" and number >= 256:
                        # Low-byte aliases selected from independent table anchors:
                        # 256->0,257->1,511->255. Those are the only high entries.
                        magnitudes = dict(GRID)[number - 256]
                    for lane, magnitude in enumerate(magnitudes):
                        position = word * 4 + lane
                        negative = (position + group + column + block) % 3 == 0
                        signs |= int(negative) << position
                        sign = -1 if negative else 1
                        if fault == "ignore_sign":
                            sign = 1
                        group_values.append(
                            h.f32(h.f32(scale * odd) * magnitude * sign)
                        )
                struct.pack_into("<I", payload, 74 + group * 4, signs)
                row.extend(group_values)
            wire.extend(payload)
        logical.append(row)
    assert len(wire) == columns * (width // 256) * 110
    return wire, logical


def audit(h):
    _, logical = weights(h, 256, 7)
    raw = inputs(h, 0, 256)
    packed = canonical(h, raw)
    records = []
    for fault in ("ignore_ninth_bit", "ignore_sign", "even_scale", "raw_input"):
        _, wrong = weights(h, 256, 7, fault)
        separations = []
        for column, row in enumerate(logical):
            expected = h.f32(math.fsum(x * y for x, y in zip(packed, row)))
            wrong_input = raw if fault == "raw_input" else packed
            actual = h.f32(math.fsum(x * y for x, y in zip(wrong_input, wrong[column])))
            budget = 0.00002 + 0.00002 * abs(expected)
            separations.append(abs(actual - expected) / budget)
        assert max(separations) > 4, fault
        records.append(
            dict(fault=fault, minimum_required=4, max_separation=max(separations))
        )
    return records


def emit(h, mode, destination):
    if mode == "audit":
        Path(destination).write_text(json.dumps(audit(h), indent=2) + "\n")
        return
    assert mode in ("tiny", "full")
    cases = [(1, 1, 256, 7)]
    if mode == "full":
        cases += [
            (1, 1, 5120, 1),
            (1, 1, 17408, 1),
            (8, 0, 256, 1),
            (8, 1, 512, 7),
            (8, 8, 256, 7),
            (16, 16, 256, 7),
            (17, 0, 256, 1),
            (17, 1, 256, 65),
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
        "iq4_nl",
        "iq3_s",
        "q3k",
    ):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    for rows, active, width, columns in cases:
        name = f"iq3_s_t{rows}_a{active}_k{width}_n{columns}"
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
                    b"".join(struct.pack("<f", v) for row in original for v in row),
                ),
                h.bank(name + ".weight", encoded),
                h.bank(
                    name + ".expected_data",
                    b"".join(struct.pack("<f", v) for v in expected),
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
  %tag = index.constant 21 : index
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
      check.expect.approximate actual(%owner) expected(%owner) atol(0.00002) rtol(0.00002) nan(same) : tensor<{padded*columns+8}xf32>,tensor<{padded*columns+8}xf32>
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
