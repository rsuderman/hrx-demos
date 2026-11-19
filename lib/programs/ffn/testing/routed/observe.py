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
# Keep exactly the gate/up definition and its declarations; append observers to
# the existing paired output buffer so inactive clearing follows the same body.
indexed = indexed[: indexed.index("kernel.def", indexed.index("kernel.def") + 1)]
indexed = replace_once(
    indexed, "@llm.experts.indexed_swiglu_q4(", "@routed.observe_gate_up("
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
Path(sys.argv[3]).write_text(router + "\n" + indexed)
