#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Emit gather checks from independently authored half wire/F32 value anchors."""

import sys
from pathlib import Path

# IEEE half wire bits and their exact real values, authored independently of
# the production extension. Include both zeros, subnormal edges and finite max.
ANCHORS = (
    (0x0000, "0.0"),
    (0x8000, "-0.0"),
    (0x0001, "0.000000059604644775390625"),
    (0x8001, "-0.000000059604644775390625"),
    (0x03FF, "0.000060975551605224609375"),
    (0x83FF, "-0.000060975551605224609375"),
    (0x0400, "0.00006103515625"),
    (0x3C00, "1.0"),
    (0xBC00, "-1.0"),
    (0x7BFF, "65504.0"),
    (0xFBFF, "-65504.0"),
    (0x3555, "0.333251953125"),
)
# Dense-only odd tails; actual root geometry; minimum/maximum H; max row count.
CASES = (
    (3, 33, 3),
    (3, 257, 3),
    (3, 3072, 128256),
    (1, 32, 1),
    (3, 32768, 3),
    (512, 32, 3),
)


def signed_byte(value):
    return str(value if value < 128 else value - 256)


def emit_case(rows, hidden, vocab):
    name = f"f16_r{rows}_h{hidden}_v{vocab}"
    extent = 2 * hidden * vocab
    ids = tuple(dict.fromkeys((0, vocab - 1, vocab // 2)))
    lines = [
        f"func.def @{name}.init(%weight: buffer, %ids: buffer) {{",
        "  %zero = index.constant 0 : index",
        "  %one = index.constant 1 : index",
        f"  %hidden = index.constant {hidden} : index",
        f"  %period = index.constant {len(ANCHORS)} : index",
    ]
    for ordinal, token in enumerate(ids):
        lines += [
            f"  %origin{ordinal} = index.constant {16 + token * hidden * 2} : offset",
            f"  %row{ordinal} = buffer.view %weight[%origin{ordinal}] : buffer -> view<{hidden * 2}xi8>",
            f"  %phase{ordinal} = index.constant {token % len(ANCHORS)} : index",
            "  scf.for %c = [%zero to %hidden step %one] {",
            f"    %sum = index.add %c, %phase{ordinal} : index",
            "    %phase = index.rem %sum, %period : index",
            "    %two = index.constant 2 : index",
            "    %lo = index.mul %c, %two : index",
            "    %hi = index.add %lo, %one : index",
        ]
        for k, (bits, _) in enumerate(ANCHORS):
            lines += [
                f"    %k{k} = index.constant {k} : index",
                f"    %at{k} = index.cmp eq, %phase, %k{k} : index",
                f"    scf.if %at{k} {{",
                f"      %low = scalar.constant {signed_byte(bits & 255)} : i8",
                f"      %high = scalar.constant {signed_byte(bits >> 8)} : i8",
                f"      view.store %low, %row{ordinal}[%lo] : i8, view<{hidden * 2}xi8>",
                f"      view.store %high, %row{ordinal}[%hi] : i8, view<{hidden * 2}xi8>",
                "    }",
            ]
        lines += ["  }"]
    lines += [
        "  %origin = index.constant 0 : offset",
        f"  %tokens = buffer.view %ids[%origin] : buffer -> view<{rows}xi32>",
    ]
    for row in range(rows):
        lines += [
            f"  %id{row} = scalar.constant {ids[row % len(ids)]} : i32",
            f"  view.store %id{row}, %tokens[{row}] : i32, view<{rows}xi32>",
        ]
    lines += [
        "  func.return",
        "}",
        f"command.def @{name}.actual() launch(%weight: buffer, %ids: buffer, %out: buffer) {{",
        f"  %rows = index.constant {rows} : index",
        f"  %hidden = index.constant {hidden} : index",
        f"  %vocab = index.constant {vocab} : index",
        "  %tag = index.constant 1 : index",
        f"  %bytes = index.constant {extent} : index",
        "  %bank = index.constant 16 : offset",
        "  %w = buffer.view %weight[%bank] : buffer -> view<[%bytes]xi8>",
        "  %o = buffer.view %out[%bank] : buffer -> view<[%rows]x[%hidden]xf32>",
        "  template.apply<@llm.embedding.gather_f32>(%rows, %hidden, %vocab, %tag, %bytes, %ids, %w, %o) : (index, index, index, index, index, buffer, view<[%bytes]xi8>, view<[%rows]x[%hidden]xf32>)",
        "  command.return",
        "}",
        f"func.def @{name}.expected(%weight: buffer, %ids: buffer, %out: buffer) {{",
        "  %zero = index.constant 0 : index",
        "  %one = index.constant 1 : index",
        f"  %period = index.constant {len(ANCHORS)} : index",
        f"  %rows = index.constant {rows} : index",
        f"  %hidden = index.constant {hidden} : index",
        "  %bank = index.constant 16 : offset",
        "  %origin = index.constant 0 : offset",
        f"  %tokens = buffer.view %ids[%origin] : buffer -> view<{rows}xi32>",
        f"  %output = buffer.view %out[%bank] : buffer -> view<{rows}x{hidden}xf32>",
        "  scf.for %r = [%zero to %rows step %one] {",
        f"    %word = view.load %tokens[%r] : view<{rows}xi32> -> i32",
        "    %token = index.cast %word : i32 to index",
        "    scf.for %c = [%zero to %hidden step %one] {",
        "      %sum = index.add %token, %c : index",
        "      %phase = index.rem %sum, %period : index",
    ]
    for k, (_, value) in enumerate(ANCHORS):
        lines += [
            f"      %k{k} = index.constant {k} : index",
            f"      %at{k} = index.cmp eq, %phase, %k{k} : index",
            f"      scf.if %at{k} {{",
            f"        %value = scalar.constant {value} : f32",
            f"        view.store %value, %output[%r, %c] : f32, view<{rows}x{hidden}xf32>",
            "      }",
        ]
    lines += [
        "    }",
        "  }",
        "  func.return",
        "}",
        f"check.scenario @{name} {{",
        "  check.trial[1](%trial: index, %entropy: check.entropy) {",
        f"    %weight = check.generate.fill value(-23) : tensor<{extent + 32}xi8>",
        f"    %ids = check.generate.fill value(-1) : tensor<{rows}xi32>",
        f"    check.generate<@{name}.init>(%weight, %ids) : (tensor<{extent + 32}xi8>, tensor<{rows}xi32>) -> ()",
        f"    %out = check.generate.fill value(-99.0) : tensor<{rows * hidden + 8}xf32>",
        f"    check.compare<@{name}.actual, @{name}.expected>(%weight, %ids, %out) : (tensor<{extent + 32}xi8>, tensor<{rows}xi32>, tensor<{rows * hidden + 8}xf32>) -> () {{",
        f"      check.expect.bitwise actual(%weight) expected(%weight) : tensor<{extent + 32}xi8>",
        f"      check.expect.bitwise actual(%ids) expected(%ids) : tensor<{rows}xi32>",
        f"      check.expect.bitwise actual(%out) expected(%out) : tensor<{rows * hidden + 8}xf32>",
        "    }",
        "  }",
        "  check.return",
        "}",
    ]
    return "\n".join(lines)


header = """// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Generated independent half wire/value, guarded dense gather checks.
template.decl @llm.embedding.gather_f32(%rows: index, %hidden: index, %vocabulary: index, %tag: index, %weight_bytes: index, %ids: buffer, %weight: view<[%weight_bytes]xi8>, %output: view<[%rows]x[%hidden]xf32>)
"""
Path(sys.argv[1]).write_text(
    header + "\n\n".join(emit_case(*case) for case in CASES) + "\n"
)
