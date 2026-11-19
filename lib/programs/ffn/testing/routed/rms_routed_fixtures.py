#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Public production RMS/routed/residual checks with independent scalar goldens.

Usage: rms_routed_fixtures.py HELPERS.py OUTPUT CAPACITY DOWN_TAG
The explicit helper path selects the maintained routed codec/oracle module.
No observed device values participate in the expected results.
"""

import importlib.util
import math
import sys
from pathlib import Path


def emit_scenario(helpers, out, name, tensors, capacity):
    # Reuse the common scenario, including bitwise inputs, output guards and
    # inactive rows. Add exact checks for channels outside the three sparse
    # down-projection witnesses, even within active rows.
    scenario = helpers.Emit()
    scenario.lines = []
    scenario.scenario(name, tensors, ["residual"], approx={"residual": (1e-3, 3e-3)})
    for line in scenario.lines:
        if line.startswith("    check.compare<"):
            for row in range(capacity):
                out.add(
                    f"    %untouched{row} = check.tensor.view %residual "
                    f"offset({4 * (row * 2048 + 2)}) : "
                    f"tensor<{capacity * 2048}xf32> -> tensor<2045xf32>"
                )
        out.lines.append(line)
        if "check.expect.approximate actual(%residual)" in line:
            for row in range(capacity):
                out.add(
                    f"      check.expect.bitwise actual(%untouched{row}) "
                    f"expected(%untouched{row}) : "
                    "tensor<2045xf32>"
                )


def generate(helpers, capacity, down_tag):
    assert capacity in (1, 8, 17), capacity
    assert down_tag in (12, 14), down_tag
    F = helpers.F
    residual = [F(0.25 * value) for value in helpers.xrow(0)]
    gamma = [F(0.025 * (1 + 0.125 * (ch % 4))) for ch in range(2048)]
    assert sum(value != 0 for value in residual) == 4
    assert len(set(gamma)) == 4 and all(value > 0 for value in gamma)
    epsilon = F(1e-6)
    squares = F(0)
    for value in residual:
        squares = F(squares + F(value * value))
    inverse = F(1 / math.sqrt(F(F(squares / 2048) + epsilon)))
    normalized = [
        F(F(value * inverse) * scale) for value, scale in zip(residual, gamma)
    ]
    assert all(math.isfinite(value) for value in normalized)
    assert sum(value != 0 for value in normalized) == 4
    assert 2**-6 <= max(map(abs, normalized)) <= 8
    # Zero logits give the documented minimum-ID tie order and exact 1/8
    # selected weights. Shared helpers own hidden math and codec interpretation.
    _, probabilities, ids, weights = helpers.route(0, 0)
    assert ids == list(range(8)) and weights == [0.125] * 8
    assert probabilities == [1 / 128] * 128
    hidden = [helpers.expert_hidden(0, expert, xs=normalized)[1] for expert in ids]
    assert all(sum(value != 0 for value in row) == 3 for row in hidden)
    expected = {}
    leaf_values = 0
    for ch in (0, 1, 2047):
        branch = F(0)
        for expert, weight, row in zip(ids, weights, hidden):
            value = helpers.dot(down_tag, expert, ch, row)
            assert math.isfinite(value) and abs(value) <= 16, value
            branch = helpers.fmadd(weight, value, branch)
            leaf_values += 1
        expected[ch] = F(residual[ch] + branch)
        assert math.isfinite(expected[ch]) and abs(expected[ch]) <= 16.25
        assert abs(branch) > 1e-3 + 3e-3 * abs(expected[ch]), (ch, branch)
    assert leaf_values == 24 and len(expected) == 3

    out = helpers.Emit()
    out.add("// Scalar F32 RMS, zero-router top8 and one residual add; no GPU oracle.")
    helpers.emit_weight_initializer(out, "rms_gate", 12, 2048, 768)
    helpers.emit_weight_initializer(out, "rms_up", 12, 2048, 768, up=True)
    helpers.emit_weight_initializer(out, "rms_down", down_tag, 768, 2048)
    out.add(
        "template.decl @llm.ffn.rms_routed(%C:index,%E:index,%epsilon:f32,"
        "%GT:index,%UT:index,%DT:index,%GB:index,%UB:index,%DB:index,"
        "%residual:buffer,%count:buffer,%gamma:view<[%E]xf32>,"
        "%router:view<128x2048xf32>,%gate:view<[%GB]xi8>,"
        "%up:view<[%UB]xi8>,%down:view<[%DB]xi8>)"
    )
    gb, db = helpers.weight_bytes(12), helpers.weight_bytes(down_tag)
    args = ["residual", "gamma", "router", "gate", "up", "down", "count"]
    counts = sorted(set((0, 1, capacity // 2, capacity, -1, capacity + 1)))
    assert len(counts) == (4 if capacity == 1 else 6)
    for active in counts:
        suffix = active if active >= 0 else "negative"
        name = f"rms_routed_q{down_tag}_c{capacity}_a{suffix}"
        out.func(name + ".init", args)
        helpers.count_init(out, active)
        # Clear only the logical RW input; retain nonzero owner guard sentinels.
        out.zero("residual", capacity * 2048)
        for row in range(capacity):
            for ch, value in enumerate(residual):
                if value:
                    out.put("residual", row * 2048 + ch, value, capacity * 2048)
        out.view("gamma", 2048)
        for ch, value in enumerate(gamma):
            out.put("gamma", ch, value, 2048)
        for key in ("gate", "up", "down"):
            out.add(f"  func.call @rms_{key}(%{key}) : (buffer)")
        out.end()
        out.command(
            name + ".actual",
            args,
            f"""  %C = index.constant {capacity} : index
  %E = index.constant 2048 : index
  %epsilon = scalar.constant {epsilon!r} : f32
  %q4 = index.constant 12 : index
  %dt = index.constant {down_tag} : index
  %GB = index.constant {gb} : index
  %DB = index.constant {db} : index
  %origin = index.constant 0 : offset
  %gamma_view = buffer.view %gamma[%origin] : buffer -> view<[%E]xf32>
  %router_view = buffer.view %router[%origin] : buffer -> view<128x2048xf32>
  %gate_view = buffer.view %gate[%origin] : buffer -> view<[%GB]xi8>
  %up_view = buffer.view %up[%origin] : buffer -> view<[%GB]xi8>
  %down_view = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>
  template.apply<@llm.ffn.rms_routed>(%C,%E,%epsilon,%q4,%q4,%dt,%GB,%GB,%DB,%residual,%count,%gamma_view,%router_view,%gate_view,%up_view,%down_view) : (index,index,f32,index,index,index,index,index,index,buffer,buffer,view<[%E]xf32>,view<128x2048xf32>,view<[%GB]xi8>,view<[%GB]xi8>,view<[%DB]xi8>)""",
        )
        out.func(name + ".expected", args)
        out.view("residual", capacity * 2048)
        for row in range(active if 0 <= active <= capacity else 0):
            for ch, value in expected.items():
                out.put("residual", row * 2048 + ch, value, capacity * 2048)
        out.end()
        tensors = [
            ("residual", capacity * 2048, "f32", "99.0"),
            ("gamma", 2048, "f32", "0.0"),
            ("router", 128 * 2048, "f32", "0.0"),
            ("gate", gb, "i8", 0),
            ("up", gb, "i8", 0),
            ("down", db, "i8", 0),
            ("count", 3, "i32", 99),
        ]
        emit_scenario(helpers, out, name, tensors, capacity)
    return "".join(out.lines)


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    helper_path, output, capacity, down_tag = sys.argv[1:]
    spec = importlib.util.spec_from_file_location("routed_oracle_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load routed helpers: {helper_path}")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    Path(output).write_text(generate(helpers, int(capacity), int(down_tag)))


if __name__ == "__main__":
    main()
