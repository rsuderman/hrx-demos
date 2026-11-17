#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent GGML-byte and staged-half numerical anchors for matrix operations."""

import math
import struct
import sys
from pathlib import Path

f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
f16 = lambda x: struct.unpack("<e", struct.pack("<e", x))[0]
SCALES = (1 / 1024, 3 / 2048, -5 / 4096, 7 / 8192)


def block_bytes(tag, channel, fill):
    block = bytearray([fill] * {12: 144, 14: 210}[tag])
    d = SCALES[channel % 4]
    if tag == 12:
        struct.pack_into("<ee", block, 0, d, d / 2)
    else:
        struct.pack_into("<e", block, 208, d)
    return block


def stage_weight(block, tag, k):
    # Deliberately scalar wire indexing, independently of production vector loads.
    if tag == 12:
        d, minimum = struct.unpack_from("<ee", block)
        scales = block[4:16]
        group = k // 32
        if group < 4:
            scale, bias = scales[group] & 63, scales[group + 4] & 63
        else:
            scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)
            bias = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
        code = (block[16 + (group // 2) * 32 + k % 32] >> (4 * (group % 2))) & 15
        ds, bs = f16(f32(d * scale)), f16(f32(-minimum * bias))
    else:
        half, within = divmod(k, 128)
        quarter, lane = divmod(within, 32)
        low = block[half * 64 + (quarter % 2) * 32 + lane]
        high = block[128 + half * 32 + lane]
        code = ((low >> (4 * (quarter // 2))) & 15) | (
            ((high >> (2 * quarter)) & 3) << 4
        )
        signed = struct.unpack_from(
            "<b", block, 192 + half * 8 + quarter * 2 + lane // 16
        )[0]
        d = struct.unpack_from("<e", block, 208)[0]
        ds = f16(f32(d * signed))
        bs = f16(f32(f32(d * signed) * -32))
    # Half FMA rounds only the final expression; half operands are exact in F64.
    return f16(float(code) * ds + bs)


def input_value(row, k, dyadic):
    if dyadic:
        return (0.125, -0.25)[row % 2]
    return f32((((k + row * 3) % 7) - 3) * f32(0.17))


def golden(tag, k_width, row, channel, dyadic, fill):
    block = block_bytes(tag, channel, fill)
    values = [f16(input_value(row, k, dyadic)) for k in range(k_width)]
    return f32(
        math.fsum(
            float(x) * stage_weight(block, tag, k % 256) for k, x in enumerate(values)
        )
    )


def emit():
    out = [
        "// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n// Generated independent anchors from fixtures.py; no production math called.\n"
    ]
    out.append(
        "template.decl @llm.projection.matrix(%T: index, %K: index, %N: index, %tag: index, %bytes: index, %count: buffer, %input: buffer, %weight: view<[%bytes]xi8>, %output: buffer)\n"
    )
    out.append(
        "template.decl @llm.projection.gated_matrix(%T: index, %K: index, %N: index, %tag: index, %bytes: index, %count: buffer, %input: buffer, %gate: view<[%bytes]xi8>, %up: view<[%bytes]xi8>, %output: buffer)\n"
    )
    for tag in ("q4k", "q5_0", "q5k", "q6k", "q8_0"):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 262144 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")
    cases = [
        ("q4_dyadic", 12, 256, 64, 17, True, False),
        ("q6_dyadic", 14, 256, 64, 17, True, False),
        ("q4_q", 12, 1024, 2048, 17, False, False),
        ("q6_v", 14, 1024, 1024, 64, False, False),
        ("q4_o", 12, 2048, 1024, 128, False, False),
        ("q6_down", 14, 3072, 1024, 17, False, False),
        ("gated", 12, 1024, 3072, 17, False, True),
    ]
    # Keep the original geometry anchors, then exercise genuinely empty and
    # partially live 64-row tiles through the same public matrix composition.
    cases = [(*case, 2) for case in cases]
    for t in (128, 512):
        for active in (0, 1, 63, 64, 65, t - 1, t):
            for kind, tag, gated in (
                ("q4", 12, False),
                ("q6", 14, False),
                ("gated", 12, True),
            ):
                cases.append(
                    (
                        f"tile_{kind}_t{t}_a{active}",
                        tag,
                        256,
                        64,
                        t,
                        True,
                        gated,
                        active,
                    )
                )
    for kind, tag, gated in (("q4", 12, False), ("q6", 14, False), ("gated", 12, True)):
        cases.append((f"zero_{kind}", tag, 256, 64, 128, True, gated, 64))
        cases.append((f"nan_{kind}", tag, 256, 64, 128, False, gated, 1))
        cases.append((f"inf_{kind}", tag, 256, 64, 128, False, gated, 1))
    # All encoded Q6 fields are finite; d=4096 with signed scale17 makes
    # the half staged scale/bias overflow. Active math remains NaN, but every
    # inactive row must be +zero, including suffixes inside the live tile.
    cases.extend(
        [
            ("overflow_q6_t128_a0", 14, 256, 64, 128, True, False, 0),
            ("overflow_q6_t128_a1", 14, 256, 64, 128, True, False, 1),
            ("overflow_q6_t512_a65", 14, 256, 64, 512, True, False, 65),
        ]
    )
    for name, tag, k, n, t, dyadic, gated, active in cases:
        zero_input = name.startswith("zero_")
        nan_input = name.startswith("nan_")
        inf_input = name.startswith("inf_")
        overflow_weight = name.startswith("overflow_")
        exceptional_result = nan_input or inf_input or overflow_weight
        tp = (t + 63) // 64 * 64
        wb = n * (k // 256) * {12: 144, 14: 210}[tag]
        out.append(
            f"""func.def @{name}.init(%input:buffer,%weight:buffer,%up:buffer,%count:buffer) {{
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %four = index.constant 4 : index
  %seven = index.constant 7 : index
  %three = index.constant 3 : index
  %K = index.constant {k} : index
  %N = index.constant {n} : index
  %blocks = index.constant {k//256} : index
  %blockbytes = index.constant {144 if tag==12 else 210} : offset
  %rowbytes = index.constant {k//256*(144 if tag==12 else 210)} : offset
  %origin = index.constant 0 : offset
  %source = buffer.view %input[%origin] : buffer -> view<{t}x{k}xf32>
  %counts = buffer.view %count[%origin] : buffer -> view<3xi32>
  %active = scalar.constant {active} : i32
  %active_rows = index.constant {active} : index
  view.store %active, %counts[%zero] : i32, view<3xi32>
  scf.for %row = [%zero to %active_rows step %one] {{
    scf.for %column = [%zero to %K step %one] {{
"""
        )
        if zero_input:
            out.append("      %value = scalar.constant 0.0 : f32\n")
        elif nan_input:
            out.append(
                "      %nan_bits = scalar.constant 2143289344 : i32\n      %value = scalar.bitcast %nan_bits : i32 to f32\n"
            )
        elif inf_input:
            # Neighboring weights are equal in these structured blocks. Opposite
            # infinities therefore require NaN in every active output channel.
            out.append("""      %parity = index.rem %column, %two : index
      %first = index.cmp eq, %parity, %zero : index
      %positive = scalar.constant 2139095040 : i32
      %negative = scalar.constant -8388608 : i32
      %bits = scf.select %first, %positive, %negative : i32
      %value = scalar.bitcast %bits : i32 to f32
""")
        elif dyadic:
            out.append("""      %parity = index.rem %row, %two : index
      %first = index.cmp eq, %parity, %zero : index
      %a = scalar.constant 0.125 : f32
      %b = scalar.constant -0.25 : f32
      %value = scf.select %first, %a, %b : f32
""")
        else:
            out.append("""      %shift = index.mul %row, %three : index
      %ordinal = index.add %column, %shift : index
      %phase = index.rem %ordinal, %seven : index
      %phase32 = index.cast %phase : index to i32
      %bias = scalar.constant 3 : i32
      %signed = scalar.subi %phase32, %bias : i32
      %value0 = scalar.sitofp %signed : i32 to f32
      %unit = scalar.constant 0.17 : f32
      %value = scalar.mulf %value0, %unit : f32
""")
        out.append(
            f"""      view.store %value, %source[%row,%column] : f32, view<{t}x{k}xf32>
    }}
  }}
  scf.for %channel = [%zero to %N step %one] {{
    %phase = index.rem %channel, %four : index
"""
        )
        for c, d in enumerate(SCALES):
            out.append(
                f"    %c{c} = index.constant {c} : index\n    %is{c} = index.cmp eq, %phase, %c{c} : index\n    %d{c} = scalar.constant {d!r} : f16\n"
            )
        out.append("""    %d01 = scf.select %is0, %d0, %d1 : f16
    %d23 = scf.select %is2, %d2, %d3 : f16
    %low = index.cmp ult, %phase, %two : index
""")
        if overflow_weight:
            out.append("    %d = scalar.constant 4096.0 : f16\n")
        else:
            out.append("    %d = scf.select %low, %d01, %d23 : f16\n")
        out.append(
            """    %rowoffset = index.scale %channel, %rowbytes : index, offset -> offset
    scf.for %block = [%zero to %blocks step %one] {
      %add = index.scale %block, %blockbytes : index, offset -> offset
      %base = index.add %rowoffset, %add : offset
"""
        )
        if tag == 12:
            out.append(
                """      %metadata = buffer.view %weight[%base] : buffer -> view<2xf16>
      %metadata_up = buffer.view %up[%base] : buffer -> view<2xf16>
      %half = scalar.constant 0.5 : f16
      %minimum = scalar.mulf %d, %half : f16
      view.store %d, %metadata[0] : f16, view<2xf16>
      view.store %minimum, %metadata[1] : f16, view<2xf16>
      view.store %d, %metadata_up[0] : f16, view<2xf16>
      view.store %minimum, %metadata_up[1] : f16, view<2xf16>
"""
            )
        else:
            out.append("""      %dadd = index.constant 208 : offset
      %dbase = index.add %base, %dadd : offset
      %metadata = buffer.view %weight[%dbase] : buffer -> view<1xf16>
      %metadata_up = buffer.view %up[%dbase] : buffer -> view<1xf16>
      view.store %d, %metadata[0] : f16, view<1xf16>
      view.store %d, %metadata_up[0] : f16, view<1xf16>
""")
        out.append("    }\n  }\n  func.return\n}\n")
        out.append(
            f"""command.def @{name}.actual() launch(%input:buffer,%weight:buffer,%up:buffer,%count:buffer,%output:buffer) {{
  %T = index.constant {t} : index
  %K = index.constant {k} : index
  %N = index.constant {n} : index
  %tag = index.constant {tag} : index
  %bytes = index.constant {wb} : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  %u = buffer.view %up[%origin] : buffer -> view<[%bytes]xi8>
"""
        )
        if gated:
            out.append(
                f"  template.apply<@llm.projection.gated_matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%u,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,view<[%bytes]xi8>,buffer)\n"
            )
        else:
            out.append(
                f"  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%bytes,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)\n"
            )
        out.append("  command.return\n}\n")
        out.append(
            f"""func.def @{name}.expected(%input:buffer,%weight:buffer,%up:buffer,%count:buffer,%output:buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %four = index.constant 4 : index
  %rows = index.constant {tp} : index
  %active_rows = index.constant {active} : index
  %nan_bits = scalar.constant 2143289344 : i32
  %nan = scalar.bitcast %nan_bits : i32 to f32
  %width = index.constant {n} : index
  %destination = buffer.view %output[%origin] : buffer -> view<{tp}x{n}xf32>
  %z = scalar.constant 0.0 : f32
  scf.for %row = [%zero to %rows step %one] {{
    scf.for %channel = [%zero to %width step %one] {{
      %phase = index.rem %channel, %four : index
      %parity = index.rem %row, %two : index
      %first = index.cmp eq, %parity, %zero : index
      %active = index.cmp ult, %row, %active_rows : index
"""
        )
        for c in range(4):
            vals = []
            for row in range(2):
                v = (
                    0.0
                    if zero_input or exceptional_result
                    else golden(tag, k, row, c, dyadic, 17)
                )
                if gated:
                    u = (
                        0.0
                        if zero_input or exceptional_result
                        else golden(tag, k, row, c, dyadic, 51)
                    )
                    silu = f32(v / (1 + math.exp(-v)))
                    v = f32(silu * u)
                vals.append(v)
            out.append(
                f"      %c{c} = index.constant {c} : index\n      %is{c} = index.cmp eq, %phase, %c{c} : index\n      %r0c{c} = scalar.constant {vals[0]!r} : f32\n      %r1c{c} = scalar.constant {vals[1]!r} : f32\n      %v{c} = scf.select %first, %r0c{c}, %r1c{c} : f32\n"
            )
        out.append("""      %v01 = scf.select %is0, %v0, %v1 : f32
      %v23 = scf.select %is2, %v2, %v3 : f32
      %low = index.cmp ult, %phase, %two : index
      %v = scf.select %low, %v01, %v23 : f32
""")
        active_value = "%nan" if exceptional_result else "%v"
        out.append(f"      %value = scf.select %active, {active_value}, %z : f32\n")
        inactive = (tp - active) * n
        inactive_view = (
            f"    %inactive = check.tensor.view %output offset({4*active*n}) : tensor<{tp*n}xf32> -> tensor<{inactive}xf32>\n"
            if inactive
            else ""
        )
        out.append(
            f"""      view.store %value, %destination[%row,%channel] : f32, view<{tp}x{n}xf32>
    }}
  }}
  func.return
}}
check.scenario @{name} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-777.0) : tensor<{t*k}xf32>
    %weight = check.generate.fill value(17) : tensor<{wb}xi8>
    %up = check.generate.fill value(51) : tensor<{wb}xi8>
    %count = check.generate.fill value(0) : tensor<3xi32>
    %owner = check.generate.fill value(-991.0) : tensor<{tp*n+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{tp*n+8}xf32> -> tensor<{tp*n}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{tp*n+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({4*(tp*n+4)}) : tensor<{tp*n+8}xf32> -> tensor<4xf32>
{inactive_view}    check.generate<@{name}.init>(%input,%weight,%up,%count) : (tensor<{t*k}xf32>,tensor<{wb}xi8>,tensor<{wb}xi8>,tensor<3xi32>) -> ()
    check.compare<@{name}.actual,@{name}.expected>(%input,%weight,%up,%count,%output) : (tensor<{t*k}xf32>,tensor<{wb}xi8>,tensor<{wb}xi8>,tensor<3xi32>,tensor<{tp*n}xf32>) -> () {{
"""
        )
        if dyadic and not gated and not exceptional_result:
            out.append(
                f"      check.expect.bitwise actual(%owner) expected(%owner) : tensor<{tp*n+8}xf32>\n"
            )
        else:
            atol, rtol = (0.0005, 0.002) if gated else (0.0001, 0.0005)
            out.append(
                f"      check.expect.approximate actual(%owner) expected(%owner) atol({atol}) rtol({rtol}) nan(same) : tensor<{tp*n+8}xf32>, tensor<{tp*n+8}xf32>\n"
            )
        if inactive:
            # Require exact inactive zero bits even when active results use the
            # existing SwiGLU tolerance or exceptional-value comparison policy.
            out.append(
                f"      check.expect.bitwise actual(%inactive) expected(%inactive) : tensor<{inactive}xf32>\n"
            )
        out.append(
            "      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>\n"
            "      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>\n"
        )
        out.append(
            f"      check.expect.bitwise actual(%input) expected(%input) : tensor<{t*k}xf32>\n      check.expect.bitwise actual(%weight) expected(%weight) : tensor<{wb}xi8>\n      check.expect.bitwise actual(%up) expected(%up) : tensor<{wb}xi8>\n      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>\n    }}\n  }}\n  check.return\n}}\n"
        )
        out.append(f"check.benchmark<@{name}> @bench_{name}\n")
    return "".join(out)


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(emit())
