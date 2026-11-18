#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Canonical matrix fallback witnesses from logical signed codes, never GPU data."""

import math
from pathlib import Path
import struct
import sys

f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
f16 = lambda x: struct.unpack("<e", struct.pack("<e", x))[0]


def input_row(row, width):
    # Every block has an exact dyadic Q8 scale and a unique extremum. Policy1's
    # scale, inverse, rounding and stored F16 scale recover these values exactly.
    values = []
    for block in range(width // 32):
        scale = (row % 3 + 1) * 2.0 ** (block % 3 - 7)
        codes = [((j * 17 + block * 3 + row * 5) % 255) - 127 for j in range(32)]
        codes[0] = 127
        values.extend(f32(code * scale) for code in codes)
    return values


def encoded_weight(tag, width, columns):
    wire, logical = bytearray(), []
    for column in range(columns):
        weights = []
        block_width = 256 if tag == 13 else 32
        for block in range(width // block_width):
            scale = (-1 if (column + block) % 2 else 1) * (column % 4 + 1) / 4096
            assert f16(scale) == scale
            if tag == 8:
                codes = [
                    ((j * 11 + column * 7 + block * 13) % 256) - 128 for j in range(32)
                ]
                codes[0], codes[-1] = -128, 127
                wire.extend(struct.pack("<e", scale))
                wire.extend(code & 255 for code in codes)
            elif tag == 6:
                codes = [
                    ((j * 7 + column * 3 + block * 5) % 32) - 16 for j in range(32)
                ]
                codes[0], codes[-1] = -16, 15
                unsigned = [code + 16 for code in codes]
                high = sum((code >> 4) << j for j, code in enumerate(unsigned))
                wire.extend(struct.pack("<eI", scale, high))
                wire.extend(
                    (unsigned[j] & 15) | ((unsigned[j + 16] & 15) << 4)
                    for j in range(16)
                )
            else:
                assert tag == 13
                code = 17 + column % 15
                codes = [code] * 256
                wire.extend(struct.pack("<ee", scale, 0.0))
                wire.extend(bytes([1] * 4 + [0] * 4 + [1] * 4))
                wire.extend(bytes([255] * 32))
                wire.extend(bytes([(code & 15) * 17] * 128))
            weights.extend(code * scale for code in codes)
        logical.append(weights)
    return wire, logical


def bank(name, data):
    return f'global.rodata.def @{name} = bytes("{data.hex()}")\n'


def copy_body(source, destination, size, kind):
    return f"""  %origin_{destination} = index.constant 0 : offset
  %zero_{destination} = index.constant 0 : index
  %one_{destination} = index.constant 1 : index
  %size_{destination} = index.constant {size} : index
  %bank_{destination} = global.load @{source} : buffer
  %src_{destination} = buffer.view %bank_{destination}[%origin_{destination}] : buffer -> view<{size}x{kind}>
  %dst_{destination} = buffer.view %{destination}[%origin_{destination}] : buffer -> view<{size}x{kind}>
  scf.for %i_{destination} = [%zero_{destination} to %size_{destination} step %one_{destination}] {{
    %v_{destination} = view.load %src_{destination}[%i_{destination}] : view<{size}x{kind}> -> {kind}
    view.store %v_{destination}, %dst_{destination}[%i_{destination}] : {kind}, view<{size}x{kind}>
  }}
"""


def emit(destination):
    out = ["// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n"]
    out.append(
        "template.decl @llm.projection.matrix(%T: index,%K: index,%N: index,%tag: index,%bytes: index,%count: buffer,%input: buffer,%weight: view<[%bytes]xi8>,%output: buffer)\n"
    )
    for tag in ("q4k", "q5_0", "q5k", "q6k", "q8_0", "q4_0", "q4_1", "iq4_xs"):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    # Both bounded T1 and T17 witnesses passed centrally before this expansion.
    cases = [
        (8, t, a, 256, n)
        for t, a, n in (
            (1, 1, 64),
            (8, 0, 1),
            (8, 1, 1),
            (8, 8, 1),
            (16, 16, 1),
            (17, 0, 1),
            (17, 1, 1),
            (17, 17, 1),
            (64, 64, 1),
        )
    ]
    cases += [
        (tag, t, a, 256, 1)
        for tag in (6, 13)
        for t, a in ((1, 1), (17, 0), (17, 1), (17, 17), (64, 1))
    ]
    for tag, rows, active, width, columns in cases:
        name = f"canonical_q{tag}_t{rows}_a{active}_k{width}_n{columns}"
        padded = rows if rows <= 16 else (rows + 63) // 64 * 64
        weight, logical = encoded_weight(tag, width, columns)
        inputs = [
            input_row(row, width) if row < active else [math.nan] * width
            for row in range(rows)
        ]
        anchors = [
            (
                f32(math.fsum(x * w for x, w in zip(inputs[row], channel)))
                if row < active
                else 0.0
            )
            for row in range(rows)
            for channel in logical
        ]
        wb = len(weight)
        tail_views, tail_checks = "", ""
        for tail, start, size in (
            ("inactive", active * columns, (rows - active) * columns),
            ("padding", rows * columns, (padded - rows) * columns),
        ):
            if size:
                tail_views += f"    %{tail} = check.tensor.view %output offset({start*4}) : tensor<{padded*columns}xf32> -> tensor<{size}xf32>\n"
                tail_checks += f"      check.expect.bitwise actual(%{tail}) expected(%{tail}) : tensor<{size}xf32>\n"
        out += [
            bank(
                name + ".input",
                b"".join(struct.pack("<f", x) for row in inputs for x in row),
            ),
            bank(name + ".weight", weight),
            bank(
                name + ".expected_data", b"".join(struct.pack("<f", x) for x in anchors)
            ),
        ]
        out.append(
            f"func.def @{name}.init(%input: buffer,%weight: buffer) {{\n"
            + copy_body(name + ".input", "input", rows * width, "f32")
            + copy_body(name + ".weight", "weight", wb, "i8")
            + "  func.return\n}\n"
        )
        out.append(
            f"func.def @{name}.expected(%count: buffer,%input: buffer,%weight: buffer,%output: buffer) {{\n"
            + copy_body(name + ".expected_data", "output", rows * columns, "f32")
            + "  func.return\n}\n"
        )
        out.append(
            f"""command.def @{name}.actual() launch(%count: buffer,%input: buffer,%weight: buffer,%output: buffer) {{
  %T = index.constant {rows} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %tag = index.constant {tag} : index
  %bytes = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial: index,%entropy: check.entropy) {{
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %input_owner = check.generate.fill value(-777.0) : tensor<{rows*width+8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{rows*width+8}xf32> -> tensor<{rows*width}xf32>
    %weight_owner = check.generate.fill value(-23) : tensor<{wb+512}xi8>
    %weight = check.tensor.view %weight_owner offset(256) : tensor<{wb+512}xi8> -> tensor<{wb}xi8>
    check.generate<@{name}.init>(%input,%weight) : (tensor<{rows*width}xf32>,tensor<{wb}xi8>) -> ()
    %owner = check.generate.fill value(-99.0) : tensor<{padded*columns+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{padded*columns+8}xf32> -> tensor<{padded*columns}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({(padded*columns+4)*4}) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>
{tail_views}
    check.compare<@{name}.actual,@{name}.expected>(%count,%input,%weight,%output) : (tensor<3xi32>,tensor<{rows*width}xf32>,tensor<{wb}xi8>,tensor<{padded*columns}xf32>) -> () {{
      check.expect.approximate actual(%owner) expected(%owner) atol(0.00001) rtol(0.00001) nan(same) : tensor<{padded*columns+8}xf32>,tensor<{padded*columns+8}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{rows*width+8}xf32>
      check.expect.bitwise actual(%weight_owner) expected(%weight_owner) : tensor<{wb+512}xi8>
{tail_checks}
    }}
  }}
  check.return
}}
"""
        )
    Path(destination).write_text("".join(out))


if __name__ == "__main__":
    emit(sys.argv[1])
