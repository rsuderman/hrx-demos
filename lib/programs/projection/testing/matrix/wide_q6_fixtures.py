#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Independent wide Q6_K wire, policy and padded-matrix witnesses.

Expected weights start as signed codes/scales, not a production decoder result.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

SCALES = (-128, -127, -65, -64, -1, 0, 1, 15, 16, 17, 63, 64, 96, 125, 126, 127)


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


def weights(h, width, columns, fault=None, late=False):
    wire, logical = bytearray(), []
    for channel in range(columns):
        row = []
        for block in range(width // 256):
            d = (-1 if (channel + block) % 2 else 1) / 65536
            if late and block < width // 256 - 1:
                d = 0.0
            assert h.f16(d) == d
            scales = [SCALES[(s + channel + block) % 16] for s in range(16)]
            ql, qh = bytearray(128), bytearray(64)
            for j in range(256):
                half, within = divmod(j, 128)
                quarter, lane = divmod(within, 32)
                code = (lane * 3 + quarter * 5 + half * 7 + channel + block) % 64 - 32
                biased = code + 32
                ql[half * 64 + quarter % 2 * 32 + lane] |= (biased & 15) << (
                    4 * (quarter // 2)
                )
                qh[half * 32 + lane] |= (biased >> 4) << (2 * quarter)
                scale = scales[j // 16]
                if fault == "ignore_high":
                    code = (biased & 15) - 32
                elif fault == "unsigned_scale":
                    scale &= 255
                elif fault == "drop_late" and block >= 68:
                    scale = 0
                value = h.f32(h.f32(d * scale) * code)
                if fault == "scaled":
                    value = h.f16(
                        (code + 32) * h.f16(d * scale) + h.f16(-32 * d * scale)
                    )
                row.append(value)
            wire.extend(ql + qh + bytes(s & 255 for s in scales) + struct.pack("<e", d))
        logical.append(row)
    assert len(wire) == columns * (width // 256) * 210
    return wire, logical


def audit(h):
    _, logical = weights(h, 27648, 7)
    source = inputs(h, 0, 27648)
    rounded = canonical(h, source)
    records = []
    for fault in (
        "raw_input",
        "ignore_high",
        "unsigned_scale",
        "drop_late",
    ):
        _, wrong = weights(h, 27648, 7, fault=fault)
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
    assert mode in ("tiny", "full", "scaled", "compatible", "late")
    cases = [(1, 1, 27648, 9)]
    if mode == "full":
        cases = [(1, 1, k, 9) for k in (3584, 17408, 17664, 21504, 22016, 27648)]
        cases += [
            (c, a, 27648, n)
            for c, a, n in (
                (1, 1, 1),
                (8, 0, 1),
                (8, 1, 7),
                (8, 5, 8),
                (8, 8, 9),
                (16, 16, 7),
            )
        ]
    if mode == "scaled":
        cases = [
            (17, a, k, 64)
            for k in (17408, 17664, 21504, 22016, 27648)
            for a in (0, 1, 9, 17)
        ]
    if mode == "compatible":
        cases = [
            (c, a, k, n)
            for k in (17408, 17664, 21504, 22016, 27648)
            for c, a, n in ((1, 1, 9), (8, 0, 1), (8, 1, 7), (8, 5, 8), (16, 16, 9))
        ]
        cases.append((1, 1, 27648, 1))
    if mode == "late":
        cases = [(1, 1, k, 9) for k in (17408, 17664, 21504, 22016, 27648)]
    out = ["// Copyright 2026 The HRX Authors\n// SPDX-License-Identifier: MIT\n"]
    out.append(
        "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%weight:view<[%bytes]xi8>,%output:buffer)\n"
    )
    out.append(
        "template.decl @llm.projection.policy_rows(%profile:index,%T:index,%K:index,%N:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%weight:view<[%bytes]xi8>,%output:buffer)\n"
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
        name = f"wide_q6_{mode}_t{rows}_a{active}_k{width}_n{columns}"
        padded = rows if rows <= 16 else (rows + 63) // 64 * 64
        encoded, logical = weights(
            h,
            width,
            columns,
            "scaled" if mode == "scaled" else None,
            late=mode == "late",
        )
        original = [
            (
                (h.input_row(r, width) if mode == "compatible" else inputs(h, r, width))
                if r < active
                else [math.nan] * width
            )
            for r in range(rows)
        ]
        rounded = [
            (
                (
                    [h.f16(x) for x in row]
                    if mode == "scaled"
                    else (row if mode == "compatible" else canonical(h, row))
                )
                if r < active
                else []
            )
            for r, row in enumerate(original)
        ]
        expected = [
            (
                h.f32(math.fsum(x * y for x, y in zip(rounded[r], w)))
                if r < active
                else (-99.0 if mode == "compatible" else 0.0)
            )
            for r in range(rows)
            for w in logical
        ]
        if mode == "scaled":
            expected += [0.0] * ((padded - rows) * columns)
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
            + h.copy_body(name + ".expected_data", "output", len(expected), "f32")
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
        invoke = "template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)"
        if mode == "compatible":
            invoke = "%profile = index.constant 0 : index\n  template.apply<@llm.projection.policy_rows>(%profile,%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)"
        # Preserve each consumed provider's established fixture policy. The
        # scaled F16 matrix reference uses an F64 sum against F32 MMA reduction;
        # fixtures.py and real_assets.py already prescribe this separate budget.
        atol, rtol = (0.0001, 0.0005) if mode == "scaled" else (0.00001, 0.00001)
        out.append(
            f"""command.def @{name}.actual() launch(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{
  %T = index.constant {rows} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %tag = index.constant 14 : index
  %bytes = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  {invoke}
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
      check.expect.approximate actual(%owner) expected(%owner) atol({atol}) rtol({rtol}) nan(same) : tensor<{padded*columns+8}xf32>,tensor<{padded*columns+8}xf32>
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
