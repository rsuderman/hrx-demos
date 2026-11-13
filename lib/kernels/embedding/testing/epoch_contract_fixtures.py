#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Expand the finite gather contract cases around independent authored oracles."""

import sys
from pathlib import Path

FORMATS = ((12, 144), (13, 176), (14, 210))
EPOCH_ROWS = (
    ("row512", 513, 0),
    ("shifted_input512", 512, 1),
    ("zero_count", 0, 0),
    ("negative_count", -1, 1),
)
ORDINARY_ROWS = (
    ("shifted_active", 2, 1),
    ("partial_position", 3, 2),
    ("zero_count", 0, 0),
    ("negative_count", -1, 0),
    ("oversized_count", 4, 0),
    ("negative_begin", 2, -1),
    ("out_of_range_begin", 2, 3),
)
PORTS = (
    "%weights: buffer, %ids: buffer, %epoch: buffer, %count: buffer, %output: buffer"
)


def wrapper(mode, tag, stride, kind, rows, vocab):
    owner = f"{mode}_contract"
    name = f"{tag}_{kind}"
    extent = stride * vocab
    constants = [("tag", tag), ("rows", rows)]
    if mode == "ordinary":
        constants.append(("hidden", 256))
    constants.extend((("vocab", vocab), ("stride", stride), ("bytes", extent)))
    if mode == "epoch":
        constants.append(("hidden", 256))
    lines = ["", f"command.def @{owner}.actual_{name}() launch({PORTS}) {{"]
    lines.extend(
        f"  %{key} = index.constant {value} : index" for key, value in constants
    )
    lines.append("  %bank = index.constant 16 : offset")
    init_args = "%tag, %rows, %vocab, %stride, %bytes, %weights, %ids, %count"
    init_types = "index, index, index, index, index, buffer, buffer, buffer"
    args = "%rows, %hidden, %vocab, %tag, %bytes, %epoch, %count, %ids, %w, %out"
    types = "index, index, index, index, index, buffer, buffer, buffer"
    expected_constants = constants[:-1]
    expected_args = "%tag, %rows, %vocab, %stride, %bytes"
    expected_types = "index, index, index, index, index"
    if mode == "ordinary":
        lines.append("  %control_origin = index.constant 12 : offset")
        init_args = "%tag, %vocab, %stride, %bytes, %weights, %ids, %epoch, %count"
        init_types = "index, index, index, index, buffer, buffer, buffer, buffer"
        args = "%rows, %hidden, %vocab, %tag, %bytes, %control_origin, %epoch, %count, %ids, %w, %out"
        types = "index, index, index, index, index, offset, buffer, buffer, buffer"
        expected_constants = [
            pair for pair in constants if pair[0] not in ("rows", "hidden")
        ]
        expected_args = "%tag, %vocab, %stride, %bytes"
        expected_types = "index, index, index, index"
    views = "view<[%bytes]xi8>, view<[%rows]x[%hidden]xf32>"
    lines.extend(
        (
            f"  kernel.launch @{owner}.init[]({init_args}) : []({init_types})",
            "  %w = buffer.view %weights[%bank] : buffer -> view<[%bytes]xi8>",
            "  %out = buffer.view %output[%bank] : buffer -> view<[%rows]x[%hidden]xf32>",
            f"  template.apply<@llm.embedding.{mode}_f32>({args}) : ({types}, {views})",
            "  command.return",
            "}",
            f"func.def @{owner}.expected_{name}({PORTS}) {{",
        )
    )
    lines.extend(
        f"  %{key} = index.constant {value} : index"
        for key, value in expected_constants
    )
    lines.extend(
        (
            f"  func.call @{owner}.expected({expected_args}, %weights, %ids, %epoch, %count, %output) : ({expected_types}, buffer, buffer, buffer, buffer, buffer) -> ()",
            "  func.return",
            "}",
        )
    )
    return "\n".join(lines) + "\n"


def scenario(mode, tag, kind, label, rows, extent, count, begin):
    owner = f"{mode}_contract"
    shapes = (
        ("weights", f"{extent + 32}xi8", -23),
        ("ids", f"{rows}xi32", -1),
        ("epoch", "8xi32", begin),
        ("count", "3xi32", count),
        ("output", f"{rows * 256 + 8}xf32", "-99.0"),
    )
    lines = ["", f"check.scenario @{mode}_{tag}_{label} {{"]
    lines.append("  check.trial[1](%trial: index, %entropy: check.entropy) {")
    lines.extend(
        f"    %{port} = check.generate.fill value({fill}) : tensor<{shape}>"
        for port, shape, fill in shapes
    )
    tensors = ", ".join(f"tensor<{shape}>" for _, shape, _ in shapes)
    lines.append(
        f"    check.compare<@{owner}.actual_{tag}_{kind}, @{owner}.expected_{tag}_{kind}>(%weights, %ids, %epoch, %count, %output) : ({tensors}) -> () {{"
    )
    for port, shape, _ in (shapes[-1], *shapes[:-1]):
        lines.append(
            f"      check.expect.bitwise actual(%{port}) expected(%{port}) : tensor<{shape}>"
        )
    lines.extend(("    }", "  }", "  check.return", "}"))
    return "\n".join(lines) + "\n"


def cases(mode):
    chunks = []
    rows = 2 if mode == "epoch" else 3
    for tag, stride in FORMATS:
        variants = (
            ("id", rows, 128257, (("high_ids", rows, 0),)),
            (
                ("row", 513, 2, EPOCH_ROWS)
                if mode == "epoch"
                else ("activity", 3, 2, ORDINARY_ROWS)
            ),
        )
        for kind, capacity, vocab, trials in variants:
            chunks.append(wrapper(mode, tag, stride, kind, capacity, vocab))
            for label, count, begin in trials:
                chunks.append(
                    scenario(
                        mode, tag, kind, label, capacity, stride * vocab, count, begin
                    )
                )
    return "".join(chunks)


core, destination = sys.argv[1:]
source = Path(core).read_text()
for mode in ("epoch", "ordinary"):
    marker = f"@@{mode.upper()}_CASES@@\n"
    if source.count(marker) != 1:
        raise ValueError(f"expected exactly one {mode} case marker")
    source = source.replace(marker, cases(mode) + ("\n" if mode == "ordinary" else ""))
Path(destination).write_text(source)
