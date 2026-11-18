#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Direct policy-QKV checks from logical codes and independent packed metadata.

The small non-quantized residual in each input group makes MMQ's original-value
sum distinguishable from the canonical quantized sum. No production decoder,
packer, captured activation or GPU result supplies the expected values.
"""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys


def load_helpers(path):
    spec = importlib.util.spec_from_file_location("canonical_fixture_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def selected_policy(profile, tag):
    return 0 if profile == 0 and tag in (12, 14) else 1


def input_row(row, width):
    factor = (-1 if row % 2 else 1) * (row + 1)
    # Amax=127/128, stable integer codes [127,40,-28,-56]. The residual 1/1024
    # changes the original sum but cannot change the nearest quantized code.
    group = (127 / 128, 321 / 1024, -28 / 128, -56 / 128)
    return [factor * group[column % 4] for column in range(width)]


def weight_parameters(h, tag, role, channel):
    sign = -1 if channel % 2 else 1
    d = h.f16((channel % 4 + role + 1) / 4093)
    code = (3, -5, 7)[role] if tag == 8 else (3, 5, 7)[role]
    if tag == 2:
        return sign * d, code, 1, 0.0, 0
    if tag == 3:
        return sign * d, code, 1, sign * (role + 1) / 32, 1
    if tag == 8:
        return d, code * sign, 1, 0.0, 0
    if tag == 14:
        return d, (1, -3, 5)[role], 3 * sign, 0.0, 0
    if tag == 23:
        # Existing shared codebook values at indices3/9/14; IQ's signed scale
        # is represented independently of its split low4/high2 wire encoding.
        return d, (-65, 13, 89)[role], 3 * sign, 0.0, 0
    return sign * d, code + (16 if tag == 13 else 0), 3, (role + 1) / 32, 1


def weights(h, tag, role, width, columns):
    wire = bytearray()
    for channel in range(columns):
        d, code, scale, minimum_d, minimum = weight_parameters(h, tag, role, channel)
        for _ in range(width // (32 if tag in (2, 3, 8) else 256)):
            if tag == 8:
                wire.extend(struct.pack("<e", d))
                wire.extend(bytes([code & 255] * 32))
            elif tag == 14:
                unsigned = code + 32
                wire.extend(bytes([(unsigned & 15) * 17] * 128))
                wire.extend(bytes([((unsigned >> 4) & 3) * 85] * 64))
                wire.extend(bytes([scale & 255] * 16))
                wire.extend(struct.pack("<e", d))
            elif tag == 23:
                unsigned = scale + 32
                high = sum((unsigned >> 4) << (2 * group) for group in range(8))
                wire.extend(struct.pack("<eH", d, high))
                wire.extend(bytes([(unsigned & 15) * 17] * 4))
                index = (3, 9, 14)[role]
                wire.extend(bytes([index * 17] * 128))
            elif tag in (2, 3):
                wire.extend(struct.pack("<e", d))
                if tag == 3:
                    # The existing independent oracle subtracts minimum_d;
                    # Q4_1 stores the corresponding additive minimum directly.
                    wire.extend(struct.pack("<e", -minimum_d))
                nibble = code + 8 if tag == 2 else code
                wire.extend(bytes([nibble * 17] * 16))
            else:
                assert tag in (12, 13)
                wire.extend(struct.pack("<ee", d, minimum_d))
                wire.extend(
                    bytes([scale] * 4 + [minimum] * 4 + [scale | (minimum << 4)] * 4)
                )
                if tag == 13:
                    wire.extend(bytes([255 if code >= 16 else 0] * 32))
                wire.extend(bytes([(code & 15) * 17] * 128))
    return wire


def dot(h, tag, role, channel, row, width, consumer_policy, active, packet_policy=None):
    packet_policy = consumer_policy if packet_policy is None else packet_policy
    values = input_row(row, 32)
    maximum = max(map(abs, values))
    # This recipe's exact dyadic scale survives all supported producer policies;
    # only the independently authored sum metadata differs in MMQ packets.
    d8 = h.f16(h.f32(maximum / 127))
    inverse = h.f32(127 / maximum)
    scaled = [h.f32(value * inverse) for value in values]
    assert min(abs(abs(value - round(value)) - 0.5) for value in scaled) > 0.01
    quantized = [
        math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
        for value in scaled
    ]
    source_sum = h.f16(math.fsum(values))
    quantized_sum = h.f16(h.f32(sum(quantized) * h.f32(maximum / 127)))
    stored_sum = source_sum if packet_policy == 0 and active > 8 else quantized_sum
    d, code, scale, minimum_d, minimum = weight_parameters(h, tag, role, channel)
    combined = h.f32(d * scale)
    if tag == 12 and consumer_policy == 0 and active > 8:
        combined = h.f16(combined)
    main = h.f32(h.f32(d8 * combined) * (sum(quantized) * code))
    correction = h.f32(h.f32(minimum_d * minimum) * stored_sum)
    return h.f32(h.f32(main - correction) * (width // 32))


def sensitivity_audit(h):
    result = []
    for tag, consumer, wrong_packet in ((12, 0, 1), (13, 1, 0)):
        correct = dot(h, tag, 1, 0, 0, 256, consumer, 17)
        wrong = dot(h, tag, 1, 0, 0, 256, consumer, 17, wrong_packet)
        budget = 0.00001 + 0.00001 * abs(correct)
        ratio = abs(correct - wrong) / budget
        assert ratio > 4, (tag, correct, wrong, budget)
        result.append(
            dict(
                tag=tag,
                consumer_policy=consumer,
                correct=correct,
                wrong_packet=wrong,
                budget=budget,
                separation_ratio=ratio,
            )
        )
    return result


def emit(h, mode, destination):
    if mode == "audit":
        Path(destination).write_text(json.dumps(sensitivity_audit(h), indent=2) + "\n")
        return
    assert mode in ("tiny", "full")
    cases = [(0, (8, 12, 14), 1, 1, 64, 32)]
    if mode == "full":
        # Expanded only after the actual tiny SOURCE/BC pair passed. Keep all
        # three policy arrangements at the capacity16 boundary and exercise
        # empty, MMVQ one/three, partial MMQ and full active extents.
        cases = [
            (profile, tags, rows, active, 64, 32)
            for profile, tags in (
                (0, (12, 14, 12)),
                (0, (8, 8, 8)),
                (0, (8, 12, 14)),
            )
            for rows, counts in (
                (1, (0, 1)),
                (16, (0, 1, 3, 16)),
                (17, (0, 1, 3, 17)),
                (64, (0, 1, 63, 64)),
            )
            for active in counts
        ]
        cases += [
            # Q4 min corrections distinguish wrong canonical packets, while
            # Q5 min corrections distinguish wrong compatible packets at the
            # SAME callsite. Q8 alone does not consume the sum metadata.
            (0, (12, 13, 8), 17, 17, 64, 32),
            (0, (12, 13, 8), 64, 64, 64, 32),
            (1, (12, 14, 8), 1, 1, 64, 32),
            (1, (12, 14, 8), 17, 17, 64, 32),
            (0, (12, 8, 14), 16, 16, 64, 32),
            (0, (12, 8, 14), 17, 17, 64, 32),
            (0, (12, 14, 8), 17, 17, 64, 32),
            (0, (12, 14, 8), 64, 63, 64, 32),
            (0, (8, 12, 14), 17, 17, 13, 7),
        ]
        # IQ remains canonical policy1 even for Llama's caller profile0. These
        # direct-helper cases are separate from ordinary matrix_qkv routing.
        cases += [
            (profile, (12, 23, 14), rows, active, 64, 64)
            for profile in (0, 1)
            for rows in (16, 17)
            for active in (0, 1, rows)
        ]
        cases += [
            (profile, (2, 3, 8), rows, active, 13, 7)
            for profile in (0, 1)
            for rows, active in ((1, 1), (17, 0), (17, 1), (17, 17))
        ]
        cases += [
            (0, (12, 2, 3), 17, 17, 13, 7),
            (0, (3, 14, 2), 17, 17, 13, 7),
        ]
        assert len(cases) == 73 and len(cases) == len(set(cases))
    out = ["// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n"]
    out.append(
        "template.decl @llm.projection.policy_qkv(%profile:index,%T:index,%K:index,%Q:index,%KV:index,%qt:index,%kt:index,%vt:index,%qb:index,%kb:index,%vb:index,%count:buffer,%input:buffer,%qw:view<[%qb]xi8>,%kw:view<[%kb]xi8>,%vw:view<[%vb]xi8>,%q:buffer,%k:buffer,%v:buffer)\n"
    )
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
        "q4_0",
        "q4_1",
    ):
        out.append(
            f"config.def @ggml.linear_{name}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{name}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    for profile, tags, rows, active, query_width, kv_width in cases:
        width = 256
        name = f"policy_qkv_p{profile}_{tags[0]}_{tags[1]}_{tags[2]}_t{rows}_a{active}_q{query_width}_kv{kv_width}"
        roles = list(zip(("q", "k", "v"), tags, (query_width, kv_width, kv_width)))
        inputs = [
            input_row(row, width) if row < active else [math.nan] * width
            for row in range(rows)
        ]
        out.append(
            h.bank(
                name + ".input",
                b"".join(struct.pack("<f", value) for row in inputs for value in row),
            )
        )
        init = h.copy_body(name + ".input", "input", rows * width, "f32")
        expected, constants, compare = "", "", ""
        types = ["tensor<3xi32>", f"tensor<{rows*width}xf32>"]
        for ordinal, (role, tag, columns) in enumerate(roles):
            wire = weights(h, tag, ordinal, width, columns)
            policy = selected_policy(profile, tag)
            out.append(h.bank(name + "." + role + "w", wire))
            init += h.copy_body(name + "." + role + "w", role + "w", len(wire), "i8")
            anchors = [
                (
                    dot(h, tag, ordinal, channel, row, width, policy, active)
                    if row < active
                    else (0.0 if policy == 1 else -991.0)
                )
                for row in range(rows)
                for channel in range(columns)
            ]
            out.append(
                h.bank(
                    name + "." + role + "_expected",
                    b"".join(struct.pack("<f", value) for value in anchors),
                )
            )
            expected += h.copy_body(
                name + "." + role + "_expected", role, rows * columns, "f32"
            )
            constants += f"  %{role}t = index.constant {tag} : index\n  %{role}b = index.constant {len(wire)} : index\n  %{role}weight = buffer.view %{role}w[%origin] : buffer -> view<[%{role}b]xi8>\n"
            types.append(f"tensor<{len(wire)}xi8>")
            compare += f"      check.expect.bitwise actual(%{role}w_owner) expected(%{role}w_owner) : tensor<{len(wire)+512}xi8>\n"
        init_args = "%input:buffer,%qw:buffer,%kw:buffer,%vw:buffer"
        out.append(
            f"func.def @{name}.init({init_args}) {{\n" + init + "  func.return\n}\n"
        )
        args = "%count:buffer,%input:buffer,%qw:buffer,%kw:buffer,%vw:buffer,%q:buffer,%k:buffer,%v:buffer"
        out.append(
            f"func.def @{name}.expected({args}) {{\n" + expected + "  func.return\n}\n"
        )
        out.append(
            f"command.def @{name}.actual() launch({args}) {{\n  %profile = index.constant {profile} : index\n  %T = index.constant {rows} : index\n  %K = index.constant {width} : index\n  %Q = index.constant {query_width} : index\n  %KV = index.constant {kv_width} : index\n  %origin = index.constant 0 : offset\n"
            + constants
            + "  template.apply<@llm.projection.policy_qkv>(%profile,%T,%K,%Q,%KV,%qt,%kt,%vt,%qb,%kb,%vb,%count,%input,%qweight,%kweight,%vweight,%q,%k,%v) : (index,index,index,index,index,index,index,index,index,index,index,buffer,buffer,view<[%qb]xi8>,view<[%kb]xi8>,view<[%vb]xi8>,buffer,buffer,buffer)\n  command.return\n}\n"
        )
        out.append(
            f"check.scenario @{name}.witness {{\n  check.trial[1](%trial:index,%entropy:check.entropy) {{\n    %count_owner = check.generate.fill value({active}) : tensor<11xi32>\n    %count = check.tensor.view %count_owner offset(16) : tensor<11xi32> -> tensor<3xi32>\n    %input_owner = check.generate.fill value(-777.0) : tensor<{rows*width+8}xf32>\n    %input = check.tensor.view %input_owner offset(16) : tensor<{rows*width+8}xf32> -> tensor<{rows*width}xf32>\n"
        )
        for ordinal, (role, tag, columns) in enumerate(roles):
            size = len(weights(h, tag, ordinal, width, columns))
            out.append(
                f"    %{role}w_owner = check.generate.fill value(-23) : tensor<{size+512}xi8>\n    %{role}w = check.tensor.view %{role}w_owner offset(256) : tensor<{size+512}xi8> -> tensor<{size}xi8>\n    %{role}_owner = check.generate.fill value(-991.0) : tensor<{rows*columns+8}xf32>\n    %{role} = check.tensor.view %{role}_owner offset(16) : tensor<{rows*columns+8}xf32> -> tensor<{rows*columns}xf32>\n    %{role}_prefix = check.tensor.view %{role}_owner offset(0) : tensor<{rows*columns+8}xf32> -> tensor<4xf32>\n    %{role}_suffix = check.tensor.view %{role}_owner offset({(rows*columns+4)*4}) : tensor<{rows*columns+8}xf32> -> tensor<4xf32>\n"
            )
            compare += f"      check.expect.approximate actual(%{role}_owner) expected(%{role}_owner) atol(0.00001) rtol(0.00001) nan(same) : tensor<{rows*columns+8}xf32>,tensor<{rows*columns+8}xf32>\n      check.expect.bitwise actual(%{role}_prefix) expected(%{role}_prefix) : tensor<4xf32>\n      check.expect.bitwise actual(%{role}_suffix) expected(%{role}_suffix) : tensor<4xf32>\n"
            if rows > active:
                out.append(
                    f"    %{role}_inactive = check.tensor.view %{role} offset({active*columns*4}) : tensor<{rows*columns}xf32> -> tensor<{(rows-active)*columns}xf32>\n"
                )
                if profile == 1 and tags == (12, 23, 14) and tag == 14:
                    # This newly covered canonical Q6 role carries nsz in its
                    # contraction. Its contract is numerical zero, not a sign
                    # bit; do not change production math to tighten that policy.
                    # Zero budgets still reject every nonzero value and NaN.
                    # All original51 cases, IQ +0 and guards remain bitwise.
                    compare += f"      check.expect.approximate actual(%{role}_inactive) expected(%{role}_inactive) atol(0.0) rtol(0.0) nan(same) : tensor<{(rows-active)*columns}xf32>,tensor<{(rows-active)*columns}xf32>\n"
                else:
                    compare += f"      check.expect.bitwise actual(%{role}_inactive) expected(%{role}_inactive) : tensor<{(rows-active)*columns}xf32>\n"
        init_types = [types[1]] + types[2:]
        types += [f"tensor<{rows*columns}xf32>" for _, _, columns in roles]
        out.append(
            f"    check.generate<@{name}.init>(%input,%qw,%kw,%vw) : ({','.join(init_types)}) -> ()\n    check.compare<@{name}.actual,@{name}.expected>(%count,%input,%qw,%kw,%vw,%q,%k,%v) : ({','.join(types)}) -> () {{\n"
            + compare
            + f"      check.expect.bitwise actual(%count_owner) expected(%count_owner) : tensor<11xi32>\n      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{rows*width+8}xf32>\n    }}\n  }}\n  check.return\n}}\n"
        )
    Path(destination).write_text("".join(out))


if __name__ == "__main__":
    helper, mode, destination = sys.argv[1:]
    emit(load_helpers(helper), mode, destination)
