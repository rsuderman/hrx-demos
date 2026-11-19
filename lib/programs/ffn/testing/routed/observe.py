#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Expose intermediate values from the actual provider bodies for isolated checks.

This is instrumentation, never an expectation generator. The independent oracle
lives in fixtures.py; production signatures and allocations stay unchanged.
"""

import sys
from pathlib import Path


def replace_once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after, 1)


router = Path(sys.argv[1]).read_text()
router = replace_once(router, "@llm.routing.softmax_top8(", "@routed.observe_route(")
router = replace_once(
    router,
    "%ids: buffer, %wts: buffer) {",
    "%ids: buffer, %wts: buffer, %logits_out: buffer, %probabilities_out: buffer) {",
)
router = replace_once(
    router,
    "  %prob = scalar.divf %lex_in, %lden : f32",
    """  %prob = scalar.divf %lex_in, %lden : f32
  scf.if %in_range {
    %observe_e = index.assume %tb [range(%tb, 0, 127)] : index
    %observe_logits = buffer.view %logits_out[%zero] : buffer -> view<[%tokens]x128xf32>
    %observe_probabilities = buffer.view %probabilities_out[%zero] : buffer -> view<[%tokens]x128xf32>
    view.store %raw, %observe_logits[%token, %observe_e] : f32, view<[%tokens]x128xf32>
    view.store %prob, %observe_probabilities[%token, %observe_e] : f32, view<[%tokens]x128xf32>
  }""",
)
router = replace_once(
    router,
    "  } else {\n    %tid_inactive =",
    """  } else {
    %observe_thread = kernel.workitem.id<x> : index
    %observe_limit = index.constant 128 : index
    %observe_writes = index.cmp ult, %observe_thread, %observe_limit : index
    scf.if %observe_writes {
      %observe_origin = index.constant 0 : offset
      %observe_zero = scalar.constant 0.0 : f32
      %observe_logits = buffer.view %logits_out[%observe_origin] : buffer -> view<[%tokens]x128xf32>
      %observe_probabilities = buffer.view %probabilities_out[%observe_origin] : buffer -> view<[%tokens]x128xf32>
      view.store %observe_zero, %observe_logits[%token, %observe_thread] : f32, view<[%tokens]x128xf32>
      view.store %observe_zero, %observe_probabilities[%token, %observe_thread] : f32, view<[%tokens]x128xf32>
    }
    %tid_inactive =""",
)
indexed = Path(sys.argv[2]).read_text()
# Instrument the actual shared production body, not the now-thin fixed-tag
# kernel wrapper. Keep the body text verbatim except observer arguments/stores.
marker = "func.def public inline @llm.experts.indexed_swiglu_body("
start = indexed.index(marker)
opening = indexed.index("{", start)
depth = 1
end = opening + 1
while depth:
    depth += (indexed[end] == "{") - (indexed[end] == "}")
    end += 1
indexed = indexed[:start] + indexed[start:end] + "\n"
indexed = replace_once(
    indexed, "@llm.experts.indexed_swiglu_body(", "@routed.observe_gate_up_body("
)
indexed = replace_once(
    indexed,
    "%w2: buffer, %y: buffer) {",
    "%w2: buffer, %y: buffer, %gate_out: buffer, %up_out: buffer) {",
)
needle = "    func.call @llm.experts.store_row(%y, %tokens, %n, %a, %out_row, %value, %c0_i32) : (buffer, index, index, index, index, f32, i32)"
indexed = replace_once(
    indexed,
    needle,
    needle
    + """
    func.call @llm.experts.store_row(%gate_out, %tokens, %n, %a, %out_row, %t0, %c0_i32) : (buffer, index, index, index, index, f32, i32)
    func.call @llm.experts.store_row(%up_out, %tokens, %n, %a, %out_row, %t1, %c0_i32) : (buffer, index, index, index, index, f32, i32)""",
)
indexed = replace_once(
    indexed,
    "      view.store %zero_inactive, %output_inactive[%a, %col_inactive] : f32, view<[%tokens]x[%n]xf32>",
    """      view.store %zero_inactive, %output_inactive[%a, %col_inactive] : f32, view<[%tokens]x[%n]xf32>
      %observe_gate = buffer.view %gate_out[%origin_inactive] : buffer -> view<[%tokens]x[%n]xf32>
      %observe_up = buffer.view %up_out[%origin_inactive] : buffer -> view<[%tokens]x[%n]xf32>
      view.store %zero_inactive, %observe_gate[%a, %col_inactive] : f32, view<[%tokens]x[%n]xf32>
      view.store %zero_inactive, %observe_up[%a, %col_inactive] : f32, view<[%tokens]x[%n]xf32>""",
)
# Both providers share declarations; each merged source module declares once.
for declaration in (
    "func.decl @llm.experts.active(%rows: index, %multiplier: index, %row: index, %count: buffer) -> (i1)\n",
    "target.decl @llm.platform.experts.wave32\n",
):
    assert router.count(declaration) == 1
    assert indexed.count(declaration) == 1
    indexed = indexed.replace(declaration, "", 1)
blocks = {12: (256, 144), 13: (256, 176), 14: (256, 210), 8: (32, 34)}
for gate in blocks:
    for up in blocks:
        symbol = (
            "routed.observe_gate_up"
            if gate == up == 12
            else f"routed.observe_gate_up_g{gate}_u{up}"
        )
        gate_row = 2048 // blocks[gate][0] * blocks[gate][1]
        up_row = 2048 // blocks[up][0] * blocks[up][1]
        indexed += f"""
kernel.def target(@llm.platform.experts.wave32) @{symbol}(%N: index, %R: index) {{
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %threads = index.constant 64 : index
  %groups = index.div %N, %two : index
  kernel.launch.config workgroups(%R, %groups, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%N0: index, %R0: index, %count: buffer, %ids: buffer, %packed: buffer, %gate: buffer, %up: buffer, %hidden: buffer, %gate_out: buffer, %up_out: buffer) {{
  %GT = index.constant {gate} : index
  %UT = index.constant {up} : index
  %GR = index.constant {gate_row} : offset
  %GE = index.constant {768 * gate_row} : offset
  %UR = index.constant {up_row} : offset
  %UE = index.constant {768 * up_row} : offset
  func.call @routed.observe_gate_up_body(%N0,%R0,%GT,%UT,%GR,%GE,%UR,%UE,%count,%ids,%packed,%gate,%up,%hidden,%gate_out,%up_out) : (index,index,index,index,offset,offset,offset,offset,buffer,buffer,buffer,buffer,buffer,buffer,buffer,buffer)
  kernel.return
}}
"""
Path(sys.argv[3]).write_text(router + "\n" + indexed)
