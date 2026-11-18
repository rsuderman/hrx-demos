#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Mixed-encoding canonical FFN arithmetic, using maintained logical recipes."""

import importlib.util
import sys
from pathlib import Path

helper, destination = sys.argv[1:]
spec = importlib.util.spec_from_file_location("fixture_helpers", helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.out.clear()
h.emit("// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n")
for tag in (2, 3, 6, 8, 12, 13, 14, 23):
    name = {
        2: "q4_0",
        3: "q4_1",
        23: "iq4_xs",
        6: "q5_0",
        8: "q8_0",
        12: "q4k",
        13: "q5k",
        14: "q6k",
    }[tag]
    h.emit(
        f"config.def @ggml.linear_{name}_q8_1_x4.token_capacity = 3 : index\nconfig.def @ggml.linear_{name}_q8_1_x4.output_capacity = 10240 : index\n"
    )
h.emit("config.def @ggml.quantize_q8_1_x4.group_capacity = 240 : index\n")
h.emit(
    "template.decl @llm.ffn.canonical_swiglu(%rows: index, %hidden: index, %ffn: index, %gt: index, %ut: index, %dt: index, %gb: index, %ub: index, %db: index, %count: buffer, %input: buffer, %gate: view<[%gb]xi8>, %up: view<[%ub]xi8>, %down: view<[%db]xi8>, %output: buffer)\n"
)
for case, (E, F, U, D) in enumerate(
    (
        (1024, 512, 12, 14),
        (1024, 512, 13, 14),
        (2560, 10240, 13, 12),
        (2560, 10240, 14, 14),
    )
):
    R = 3
    G = E * F // 256 * 144
    UB = E * F // 256 * {12: 144, 13: 176, 14: 210}[U]
    DB = E * F // 256 * {12: 144, 14: 210}[D]
    initial, _, sums = h.ffn_anchors(R, E, F)
    result = [[h.f32(sums[r][k] - initial[r][k]) for k in range(4)] for r in range(R)]
    body = h.prefix() + h.input_body(R, E, True) + "\n"
    # Gate/up rows range over F, while down rows range over E.
    # Constructed F<E must still initialize every down output row.
    body += f"  %intermediate = index.constant {F} : index\n  %gate_row = index.cmp ult, %channel, %intermediate : index\n  scf.if %gate_row {{\n"
    body += h.weight_body(12, E, "gate", "gate", "g")
    body += h.weight_body(U, E, "up", "up", "u") + "  }\n"
    body += f"  %hidden = index.constant {E} : index\n  %small = index.cmp ult, %channel, %hidden : index\n  scf.if %small {{\n"
    body += h.weight_body(D, F, "down", "down", "d") + "  }\n"
    h.kernel(
        f"spark_ffn.init{case}",
        max(E, F),
        "%input: buffer, %gate: buffer, %up: buffer, %down: buffer",
        body,
    )
    h.reference(
        f"spark_ffn.init{case}",
        max(E, F),
        "%input: buffer, %gate: buffer, %up: buffer, %down: buffer",
        f"    func.call @spark_ffn.init{case}.body(%channel, %input, %gate, %up, %down) : (index, buffer, buffer, buffer, buffer) -> ()",
    )
    h.kernel(
        f"test.expected.spark_ffn{case}",
        E,
        "%output: buffer",
        h.prefix() + h.store_pattern(R, E, "output", result, "out"),
    )
    h.reference(
        f"spark_ffn{case}",
        E,
        "%input: buffer, %gate: buffer, %up: buffer, %down: buffer, %count: buffer, %output: buffer",
        f"    func.call @test.expected.spark_ffn{case}.body(%channel, %output) : (index, buffer) -> ()",
    )
    h.emit(
        f"""command.def @spark_ffn.actual{case}() launch(%input: buffer, %gate: buffer, %up: buffer, %down: buffer, %count: buffer, %output: buffer) {{
  %rows = index.constant {R} : index
  %hidden = index.constant {E} : index
  %ffn = index.constant {F} : index
  %gt = index.constant 12 : index
  %ut = index.constant {U} : index
  %dt = index.constant {D} : index
  %gb = index.constant {G} : index
  %ub = index.constant {UB} : index
  %db = index.constant {DB} : index
  %origin = index.constant 0 : offset
  %g = buffer.view %gate[%origin] : buffer -> view<[%gb]xi8>
  %u = buffer.view %up[%origin] : buffer -> view<[%ub]xi8>
  %d = buffer.view %down[%origin] : buffer -> view<[%db]xi8>
  template.apply<@llm.ffn.canonical_swiglu>(%rows, %hidden, %ffn, %gt, %ut, %dt, %gb, %ub, %db, %count, %input, %g, %u, %d, %output) : (index, index, index, index, index, index, index, index, index, buffer, buffer, view<[%gb]xi8>, view<[%ub]xi8>, view<[%db]xi8>, buffer)
  command.return
}}
check.scenario @spark_ffn.case{case} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input = check.generate.fill value(-77.0) : tensor<{R*E}xf32>
    %gate = check.generate.fill value(0) : tensor<{G}xi8>
    %up = check.generate.fill value(0) : tensor<{UB}xi8>
    %down = check.generate.fill value(0) : tensor<{DB}xi8>
    %count = check.generate.fill value(3) : tensor<3xi32>
    %owner = check.generate.fill value(-99.0) : tensor<{R*E+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{R*E+8}xf32> -> tensor<{R*E}xf32>
    check.generate<@reference.spark_ffn.init{case}>(%input, %gate, %up, %down) : (tensor<{R*E}xf32>, tensor<{G}xi8>, tensor<{UB}xi8>, tensor<{DB}xi8>) -> ()
    check.compare<@spark_ffn.actual{case}, @reference.spark_ffn{case}>(%input, %gate, %up, %down, %count, %output) : (tensor<{R*E}xf32>, tensor<{G}xi8>, tensor<{UB}xi8>, tensor<{DB}xi8>, tensor<3xi32>, tensor<{R*E}xf32>) -> () {{
      check.expect.approximate actual(%owner) expected(%owner) atol(0.0001) rtol(0.0001) nan(same) : tensor<{R*E+8}xf32>, tensor<{R*E+8}xf32>
      check.expect.bitwise actual(%input) expected(%input) : tensor<{R*E}xf32>
      check.expect.bitwise actual(%gate) expected(%gate) : tensor<{G}xi8>
      check.expect.bitwise actual(%up) expected(%up) : tensor<{UB}xi8>
      check.expect.bitwise actual(%down) expected(%down) : tensor<{DB}xi8>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
    }}
  }}
  check.return
}}
"""
    )
Path(destination).write_text("".join(h.out))
