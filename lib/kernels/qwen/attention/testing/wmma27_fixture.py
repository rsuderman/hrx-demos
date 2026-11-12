# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Reuse the independent scalar GQA oracle at the admitted 27B geometry.

This changes fixture dimensions, plane offsets, and the GQA ratio only. Keeping
one oracle source ensures the retained 0.8B and 27B cases test the same algebra.
The production kernels and configuration generator are never inputs.
"""

import pathlib
import sys
import math
import struct

source = pathlib.Path(sys.argv[1]).read_text()
for before, after in {
    "4x2048xf32": "4x6144xf32",
    "262208xf16": "524352xf16",
    "%eight = index.constant 8 : index": "%eight = index.constant 24 : index",
    "%kv_width = index.constant 512 : index": "%kv_width = index.constant 1024 : index\n  %kv_heads = index.constant 4 : index",
    "%key_base = index.constant 131104 : index": "%key_base = index.constant 262176 : index",
    "%value_base = index.constant 196640 : index": "%value_base = index.constant 393248 : index",
    "scf.for %h = [%zero to %two step %one]": "scf.for %h = [%zero to %kv_heads step %one]",
    "%kv_head = index.div %h, %four : index": "%queries_per_kv = index.constant 6 : index\n      %kv_head = index.div %h, %queries_per_kv : index",
    "%layer = index.constant 262144 : offset": "%layer = index.constant 524288 : offset",
    "offset(16384)": "offset(49152)",
    "tensor<4096xf32>": "tensor<12288xf32>",
    "offset(24576)": "offset(73728)",
    "tensor<2048xf32>": "tensor<6144xf32>",
}.items():
    if before not in source:
        raise ValueError(f"oracle fixture changed; review 27B adaptation: {before}")
    source = source.replace(before, after)

# All scores are 0 or q (q=0,1,2), all V and gates are strictly positive:
# conditioning=sum|p*v|/|sum(p*v)|=1. Full-page/tail maxima are unchanged, so
# online rescale is exactly 0/1. gamma64 bounds the longest F16 positive dot.
# Four additional F16 roundings cover probability, block combination, inverse,
# and normalized result. 2e-5 covers the two shifted/unshifted F32 exp evaluations;
# gamma96(F32) covers a more conservative denominator reduction than this case.
# This is a finite positive-fixture bound, not a general WMMA relative guarantee.
u = 2**-11
gamma64 = 64 * u / (1 - 64 * u)
f32_u = 2**-24
gamma96 = 96 * f32_u / (1 - 96 * f32_u)
assert (1 + gamma64) * (1 + u) ** 4 * (1 + 2e-5) / (1 - gamma96) - 1 < 0.035
source = source.replace("atol(0.002) rtol(0.002)", "atol(0.00002) rtol(0.035)")


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def policy_value(position, row, nonzero, head, lane):
    """Finite scalar rounding anchors, independent of layouts/fragments.

    RDNA3 ISA section7.9 documents WMMA using DOT instructions with RNE;
    fdot2_f16_f16 adds two products to an F16 accumulator. Thus one scalar
    product must not be prematurely rounded before its paired product.
    Adjacent-pair order is a finite candidate, not a universal WMMA ordering
    assertion. Every head/lane and boundary checks that candidate separately.
    The separately retained ideal oracle supplies the justified error budget.
    """
    context = position + row + 1
    query = (head % 2 + 1) * nonzero
    maximum = query if context > 1 else 0
    probability = [half(math.exp(query * (t % 2) - maximum)) for t in range(context)]
    amplitude = (head // 6 + 1) * (lane % 4 + 1) * 0.125
    values = [amplitude * (1 + (t % 3) * 0.5) for t in range(context)]
    full, tail = 0.0, 0.0
    # Primary operation contracts, rather than observed output corrections:
    # https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
    # https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html
    for t in range(0, min(context, 64), 2):
        full = half(
            full
            + sum(probability[j] * values[j] for j in range(t, min(t + 2, context, 64)))
        )
    for t in range(64, context, 2):
        tail = half(
            tail
            + sum(probability[j] * values[j] for j in range(t, min(t + 2, context)))
        )
    numerator = half(full + tail)
    inverse = half(1 / sum(math.exp(query * (t % 2) - maximum) for t in range(context)))
    return half(numerator * inverse) * ((head % 4 + 1) * 0.25)


args = "%epoch: buffer, %rows: buffer, %query: buffer, %gate: buffer, %arena: buffer, %output: buffer"
for suffix, count, position, nonzero in [
    ("n2_p0_q1", 2, 0, 1),
    ("n3_p63_q0", 3, 63, 0),
    ("n3_p63_q1", 3, 63, 1),
]:
    body = f"""func.def @wmma_policy_reference_{suffix}({args}) {{
  %count = index.constant {count} : index
  %position = index.constant {position} : index
  %nonzero = index.constant {nonzero} : index
  func.call @wmma_init(%count, %position, %nonzero, %epoch, %rows, %query, %gate, %arena) : (index, index, index, buffer, buffer, buffer, buffer, buffer) -> ()
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %four = index.constant 4 : index
  %width = index.constant 256 : index
  %origin = index.constant 0 : offset
  %out = buffer.view %output[%origin] : buffer -> view<4x6144xf32>
"""
    for row in range(count):
        for head in range(24):
            prefix = f"r{row}h{head}"
            body += f"  %{prefix}row = index.constant {row} : index\n  %{prefix}base = index.constant {head*256} : index\n"
            for lane in range(4):
                value = policy_value(position, row, nonzero, head, lane)
                body += f"  %{prefix}v{lane} = scalar.constant {value:.17f} : f32\n"
            body += f"  scf.for %{prefix}c = [%zero to %width step %one] {{\n    %{prefix}mod = index.rem %{prefix}c, %four : index\n"
            for lane in range(3):
                body += f"    %{prefix}i{lane} = index.constant {lane} : index\n    %{prefix}hit{lane} = index.cmp eq, %{prefix}mod, %{prefix}i{lane} : index\n"
            body += f"""    %{prefix}a = scf.select %{prefix}hit2, %{prefix}v2, %{prefix}v3 : f32
    %{prefix}b = scf.select %{prefix}hit1, %{prefix}v1, %{prefix}a : f32
    %{prefix}v = scf.select %{prefix}hit0, %{prefix}v0, %{prefix}b : f32
    %{prefix}column = index.add %{prefix}base, %{prefix}c : index
    view.store %{prefix}v, %out[%{prefix}row, %{prefix}column] : f32, view<4x6144xf32>
  }}
"""
    body += "  func.return\n}\n"
    start = source.index("check.scenario @wmma_" + suffix)
    end = source.index("\n}", start) + 2
    scenario = (
        source[start:end]
        .replace("@wmma_" + suffix, "@wmma_policy_" + suffix)
        .replace("@wmma_reference_" + suffix, "@wmma_policy_reference_" + suffix)
        .replace("atol(0.00002) rtol(0.035)", "atol(0.000002) rtol(0.000002)")
    )
    source += "\n" + body + scenario + "\n"

pathlib.Path(sys.argv[2]).write_text(source)
