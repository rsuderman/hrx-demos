#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Independent IQ4_NL canonical projection and nonlinear-codebook witnesses.

Weights are encoded from logical nibble indices and signed half scales. Expected
products independently model canonical Q8 activation rounding, not the production
decoder or donor raw-F32-input contraction.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

CODEBOOK = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
SCALES = (1 / 32, -1 / 64, 0.0, 3 / 128)


def helpers(path):
    spec = importlib.util.spec_from_file_location("canonical_fixture_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(h, row, width, anchors=False):
    if anchors:
        # Four disjoint blocks exercise positive, negative, zero and fractional
        # scales; every output channel chooses a different codebook entry.
        values = [0.0] * width
        values[(row % 4) * 32] = 1.0
        return values
    factor = (row % 3 + 1) * (-1 if row % 2 else 1)
    low = (127 / 128, 321 / 1024, -28 / 128, -56 / 128)
    high = (53 / 128, -173 / 1024, -11 / 128, 31 / 128)
    return [h.f32(factor * (low if j % 32 < 16 else high)[j % 4]) for j in range(width)]


def canonical(h, values):
    result = []
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        d = h.f32(max(map(abs, block)) / 127)
        inverse = h.f32(1 / d) if d else 0.0
        scaled = [h.f32(value * inverse) for value in block]
        codes = [math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5) for x in scaled]
        assert all(-127 <= x <= 127 for x in codes)
        result.extend(h.f32(code * h.f16(d)) for code in codes)
    return result


def weights(h, width, columns, fault=None):
    wire, logical = bytearray(), []
    for column in range(columns):
        row = []
        for block in range(width // 32):
            d = SCALES[block % len(SCALES)] * (1 + column // 16)
            assert h.f16(d) == d
            indices = [(j * 5 + (j // 16) * 7 + column) % 16 for j in range(32)]
            wire.extend(struct.pack("<e", d))
            wire.extend(indices[j] | (indices[j + 16] << 4) for j in range(16))
            if fault == "swap_nibbles":
                indices = indices[16:] + indices[:16]
            scale = abs(d) if fault == "unsigned_scale" else d
            row.extend(
                h.f32(
                    scale
                    * (index - 8 if fault == "linear_codebook" else CODEBOOK[index])
                )
                for index in indices
            )
        logical.append(row)
    assert len(wire) == columns * width // 32 * 18
    return wire, logical


def audit(h):
    _, logical = weights(h, 128, 16)
    raw = inputs(h, 0, 128)
    rounded = canonical(h, raw)
    records = []
    for fault in ("raw_input", "swap_nibbles", "unsigned_scale", "linear_codebook"):
        _, wrong_weights = weights(h, 128, 16, fault=fault)
        comparisons = []
        for channel, row in enumerate(logical):
            expected = h.f32(math.fsum(x * w for x, w in zip(rounded, row)))
            wrong_input = raw if fault == "raw_input" else rounded
            wrong = h.f32(
                math.fsum(x * w for x, w in zip(wrong_input, wrong_weights[channel]))
            )
            budget = 0.00001 + 0.00001 * abs(expected)
            comparisons.append(
                dict(
                    channel=channel,
                    expected=expected,
                    wrong=wrong,
                    separation=abs(expected - wrong) / budget,
                )
            )
        assert max(record["separation"] for record in comparisons) > 4, fault
        records.append(dict(fault=fault, cases=comparisons))
    return records


def emit(h, mode, destination):
    if mode == "audit":
        Path(destination).write_text(json.dumps(audit(h), indent=2) + "\n")
        return
    assert mode in ("tiny", "full")
    # (rows, active rows, width, output columns, one-hot codebook anchors).
    cases = [(4, 4, 128, 16, True), (1, 1, 128, 7, False)]
    if mode == "full":
        cases += [
            (8, 0, 128, 1, False),
            (8, 1, 256, 7, False),
            (8, 8, 128, 65, False),
            (16, 16, 256, 7, False),
            (17, 0, 128, 1, False),
            (17, 1, 128, 65, False),
            (17, 17, 256, 7, False),
            (65, 17, 128, 65, False),
            (1, 1, 5120, 7, False),
            (1, 1, 17408, 7, False),
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
    for rows, active, width, columns, anchors in cases:
        name = f"iq4_nl_t{rows}_a{active}_k{width}_n{columns}"
        padded = rows if rows <= 16 else (rows + 63) // 64 * 64
        encoded, logical = weights(h, width, columns)
        original = [
            inputs(h, row, width, anchors) if row < active else [math.nan] * width
            for row in range(rows)
        ]
        rounded = [
            canonical(h, row) if i < active else [] for i, row in enumerate(original)
        ]
        expected = [
            (
                h.f32(math.fsum(x * w for x, w in zip(rounded[row], channel)))
                if row < active
                else 0.0
            )
            for row in range(rows)
            for channel in logical
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
  %tag = index.constant 20 : index
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
