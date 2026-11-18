#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""IQ4_XS anchors from logical codebook indices/scales and canonical rounding.

No production decoder, packer or captured model output supplies expected values.
Existing fixture helpers provide only rodata/copy scaffolding and F32/F16 casts.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

CODEBOOK = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
SCALES = (-32, -17, -1, 0, 1, 15, 16, 31)


def helpers(path):
    spec = importlib.util.spec_from_file_location("canonical_fixture_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(h, row, width):
    # 321/1024 rounds to 40 canonical codes but is not exactly representable by
    # that packet; a raw-F32-input substitute therefore has a different answer.
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
        stored_scale = h.f16(d)
        result.extend(h.f32(x * stored_scale) for x in codes)
    return result


def weights(h, width, columns, fault=None):
    wire, logical = bytearray(), []
    for column in range(columns):
        row = []
        for block in range(width // 256):
            # Exact dyadic block scales alternate sign and differ per block/row.
            d = (-1 if (column + block) % 2 else 1) * (column % 3 + 1) / 65536
            assert h.f16(d) == d
            scales = [SCALES[(g + block + column) % 8] for g in range(8)]
            unsigned = [scale + 32 for scale in scales]
            high = sum(
                (scale >> 4) << (2 * group) for group, scale in enumerate(unsigned)
            )
            low = [
                unsigned[2 * j] % 16 | ((unsigned[2 * j + 1] % 16) << 4)
                for j in range(4)
            ]
            wire.extend(struct.pack("<eH", d, high))
            wire.extend(low)
            for group, scale in enumerate(scales):
                indices = [
                    (j * 5 + (j // 16) * 7 + group * 3 + block * 7 + column) % 16
                    for j in range(32)
                ]
                wire.extend(indices[j] | (indices[j + 16] << 4) for j in range(16))
                if fault == "swap_nibbles":
                    indices = indices[16:] + indices[:16]
                if fault == "ignore_scale_high":
                    scale = (scale + 32) % 16 - 32
                row.extend(
                    h.f32(h.f32(d * scale) * CODEBOOK[index]) for index in indices
                )
        logical.append(row)
    assert len(wire) == columns * width // 256 * 136
    return wire, logical


def audit(h):
    _, w = weights(h, 256, 9)
    x = inputs(h, 0, 256)
    q = canonical(h, x)
    records = []
    for fault in ("raw_input", "swap_nibbles", "ignore_scale_high"):
        _, wrong_weights = weights(h, 256, 9, fault=fault)
        cases = []
        for column, row in enumerate(w):
            expected = h.f32(math.fsum(a * b for a, b in zip(q, row)))
            wrong_input = x if fault == "raw_input" else q
            wrong = h.f32(
                math.fsum(a * b for a, b in zip(wrong_input, wrong_weights[column]))
            )
            budget = 0.00001 + 0.00001 * abs(expected)
            cases.append(
                dict(
                    column=column,
                    canonical=expected,
                    wrong=wrong,
                    separation=abs(expected - wrong) / budget,
                )
            )
        assert max(record["separation"] for record in cases) > 4, fault
        records.append(dict(fault=fault, cases=cases))
    return records


def emit_mixed(h):
    """Actual matrix_qkv: preserve Q4/Q6 T17 scaled providers beside IQ fallback."""
    out = [
        "template.decl @llm.projection.matrix_qkv(%T:index,%K:index,%Q:index,%KV:index,%qt:index,%kt:index,%vt:index,%QB:index,%KB:index,%VB:index,%count:buffer,%input:buffer,%qw:view<[%QB]xi8>,%kw:view<[%KB]xi8>,%vw:view<[%VB]xi8>,%q:buffer,%k:buffer,%v:buffer)\n"
    ]
    width, columns = 256, 64
    q4 = (
        struct.pack("<ee", 0.5, 0.0)
        + bytes([3] * 4 + [0] * 4 + [3] * 4)
        + bytes([17] * 128)
    ) * columns
    q6 = (
        bytes([17] * 128)
        + bytes([170] * 64)
        + bytes([1] * 16)
        + struct.pack("<e", 0.25)
    ) * columns
    iq, iq_logical = weights(h, width, columns)
    for rows in (16, 17):
        for active in (0, 1, rows):
            name = f"matrix_qkv_iq_t{rows}_a{active}"
            padded = rows if rows <= 16 else (rows + 63) // 64 * 64
            source = [
                inputs(h, r, width) if r < active else [math.nan] * width
                for r in range(rows)
            ]
            rounded = [
                canonical(h, row) if r < active else [] for r, row in enumerate(source)
            ]
            out.append(
                h.bank(
                    name + ".input",
                    b"".join(struct.pack("<f", x) for row in source for x in row),
                )
            )
            init = h.copy_body(name + ".input", "input", rows * width, "f32")
            expected, compare, views, constants = "", "", "", ""
            types = ["tensor<3xi32>", f"tensor<{rows*width}xf32>"]
            for role, tag, wire, value in (
                ("q", 12, q4, 1.5),
                ("k", 23, iq, None),
                ("v", 14, q6, 0.25),
            ):
                out.append(h.bank(name + "." + role + "w", wire))
                init += h.copy_body(
                    name + "." + role + "w", role + "w", len(wire), "i8"
                )
                anchors = []
                for row in range(rows):
                    for channel in range(columns):
                        if row >= active:
                            anchors.append(0.0)
                        elif tag == 23:
                            anchors.append(
                                h.f32(
                                    math.fsum(
                                        x * w
                                        for x, w in zip(
                                            rounded[row], iq_logical[channel]
                                        )
                                    )
                                )
                            )
                        else:
                            # Both tiny weights and inputs are exactly F16, so
                            # T17's existing scaled route retains raw input;
                            # T16's existing canonical route rounds the residual.
                            x = source[row] if rows > 16 else rounded[row]
                            anchors.append(h.f32(math.fsum(x) * value))
                out.append(
                    h.bank(
                        name + "." + role + "_expected",
                        b"".join(struct.pack("<f", x) for x in anchors),
                    )
                )
                expected += h.copy_body(
                    name + "." + role + "_expected", role, rows * columns, "f32"
                )
                constants += f"  %{role}t = index.constant {tag} : index\n  %{role}b = index.constant {len(wire)} : index\n  %{role}weight = buffer.view %{role}w[%origin] : buffer -> view<[%{role}b]xi8>\n"
                types.append(f"tensor<{len(wire)}xi8>")
                views += f"    %{role}w_owner = check.generate.fill value(-23) : tensor<{len(wire)+512}xi8>\n    %{role}w = check.tensor.view %{role}w_owner offset(256) : tensor<{len(wire)+512}xi8> -> tensor<{len(wire)}xi8>\n    %{role}_owner = check.generate.fill value(-991.0) : tensor<{padded*columns+8}xf32>\n    %{role} = check.tensor.view %{role}_owner offset(16) : tensor<{padded*columns+8}xf32> -> tensor<{padded*columns}xf32>\n    %{role}_prefix = check.tensor.view %{role}_owner offset(0) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>\n    %{role}_suffix = check.tensor.view %{role}_owner offset({(padded*columns+4)*4}) : tensor<{padded*columns+8}xf32> -> tensor<4xf32>\n"
                compare += f"      check.expect.approximate actual(%{role}_owner) expected(%{role}_owner) atol(0.00001) rtol(0.00001) nan(same) : tensor<{padded*columns+8}xf32>,tensor<{padded*columns+8}xf32>\n      check.expect.bitwise actual(%{role}_prefix) expected(%{role}_prefix) : tensor<4xf32>\n      check.expect.bitwise actual(%{role}_suffix) expected(%{role}_suffix) : tensor<4xf32>\n      check.expect.bitwise actual(%{role}w_owner) expected(%{role}w_owner) : tensor<{len(wire)+512}xi8>\n"
                if rows > active:
                    views += f"    %{role}_inactive = check.tensor.view %{role} offset({active*columns*4}) : tensor<{padded*columns}xf32> -> tensor<{(rows-active)*columns}xf32>\n"
                    compare += f"      check.expect.bitwise actual(%{role}_inactive) expected(%{role}_inactive) : tensor<{(rows-active)*columns}xf32>\n"
                if padded > rows:
                    # Scaled Q4/Q6 write zero through padded rows, while the IQ
                    # canonical fallback only owns logical rows.
                    padding = 0.0 if tag != 23 else -991.0
                    fill = f"  %pad_step_{role} = index.constant 1 : index\n  %pad_begin_{role} = index.constant {rows*columns} : index\n  %pad_end_{role} = index.constant {padded*columns} : index\n  %pad_{role} = scalar.constant {padding} : f32\n  %pad_origin_{role} = index.constant 0 : offset\n  %padview_{role} = buffer.view %{role}[%pad_origin_{role}] : buffer -> view<{padded*columns}xf32>\n  scf.for %pad_i_{role} = [%pad_begin_{role} to %pad_end_{role} step %pad_step_{role}] {{ view.store %pad_{role},%padview_{role}[%pad_i_{role}] : f32,view<{padded*columns}xf32> }}\n"
                    expected += fill
                    views += f"    %{role}_padding = check.tensor.view %{role} offset({rows*columns*4}) : tensor<{padded*columns}xf32> -> tensor<{(padded-rows)*columns}xf32>\n"
                    compare += f"      check.expect.bitwise actual(%{role}_padding) expected(%{role}_padding) : tensor<{(padded-rows)*columns}xf32>\n"
            out.append(
                f"func.def @{name}.init(%input:buffer,%qw:buffer,%kw:buffer,%vw:buffer) {{\n"
                + init
                + "  func.return\n}\n"
            )
            args = "%count:buffer,%input:buffer,%qw:buffer,%kw:buffer,%vw:buffer,%q:buffer,%k:buffer,%v:buffer"
            out.append(
                f"func.def @{name}.expected({args}) {{\n"
                + expected
                + "  func.return\n}\n"
            )
            out.append(
                f"command.def @{name}.actual() launch({args}) {{\n  %T = index.constant {rows} : index\n  %K = index.constant {width} : index\n  %N = index.constant {columns} : index\n  %origin = index.constant 0 : offset\n"
                + constants
                + "  template.apply<@llm.projection.matrix_qkv>(%T,%K,%N,%N,%qt,%kt,%vt,%qb,%kb,%vb,%count,%input,%qweight,%kweight,%vweight,%q,%k,%v) : (index,index,index,index,index,index,index,index,index,index,buffer,buffer,view<[%qb]xi8>,view<[%kb]xi8>,view<[%vb]xi8>,buffer,buffer,buffer)\n  command.return\n}\n"
            )
            initial_types = types[1:]
            types += [f"tensor<{padded*columns}xf32>"] * 3
            out.append(
                f"check.scenario @{name}.witness {{\n  check.trial[1](%trial:index,%entropy:check.entropy) {{\n    %count_owner = check.generate.fill value({active}) : tensor<11xi32>\n    %count = check.tensor.view %count_owner offset(16) : tensor<11xi32> -> tensor<3xi32>\n    %input_owner = check.generate.fill value(-777.0) : tensor<{rows*width+8}xf32>\n    %input = check.tensor.view %input_owner offset(16) : tensor<{rows*width+8}xf32> -> tensor<{rows*width}xf32>\n"
                + views
                + f"    check.generate<@{name}.init>(%input,%qw,%kw,%vw) : ({','.join(initial_types)}) -> ()\n    check.compare<@{name}.actual,@{name}.expected>(%count,%input,%qw,%kw,%vw,%q,%k,%v) : ({','.join(types)}) -> () {{\n"
                + compare
                + f"      check.expect.bitwise actual(%count_owner) expected(%count_owner) : tensor<11xi32>\n      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{rows*width+8}xf32>\n    }}\n  }}\n  check.return\n}}\n"
            )
    return out


def emit(h, mode, destination):
    if mode == "audit":
        Path(destination).write_text(json.dumps(audit(h), indent=2) + "\n")
        return
    assert mode in ("tiny", "full")
    cases = [(1, 1, 256, 9)]
    if mode == "full":
        cases = [
            (1, 1, 256, 9),
            (1, 1, 2048, 9),
            (1, 1, 3072, 9),
            (8, 0, 2048, 1),
            (8, 1, 3072, 9),
            (8, 8, 2048, 9),
            (16, 16, 2048, 9),
            (17, 0, 3072, 1),
            (17, 1, 2048, 9),
            (17, 17, 3072, 9),
            (65, 17, 256, 9),
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
        "iq4_nl",
        "iq3_s",
        "q3k",
        "q4_0",
        "q4_1",
    ):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    for rows, count, width, columns in cases:
        assert 0 <= count <= rows
        active = count
        name = f"iq4_xs_t{rows}_a{count}_k{width}_n{columns}"
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
  %tag = index.constant 23 : index
  %bytes = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %count_owner = check.generate.fill value({count}) : tensor<11xi32>
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
    if mode == "full":
        out.extend(emit_mixed(h))
    Path(destination).write_text("".join(out))


if __name__ == "__main__":
    helper_path, mode, destination = sys.argv[1:]
    emit(helpers(helper_path), mode, destination)
