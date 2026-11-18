#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent adjacent-pair H64 anchors with exact 32-factor ownership."""

import math
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rodata(name, values, kind="f"):
    return (
        f'global.rodata.def @{name} = bytes("'
        + b"".join(struct.pack("<" + kind, v) for v in values).hex()
        + '")'
    )


def emit():
    step = f32(500000.0 ** (-2.0 / 64))
    factors = [f32(1 + (i % 5) / 4) for i in range(32)]
    lines = [
        f"""config.def @llama.model.query_heads = 32 : index
config.def @llama.model.kv_heads = 8 : index
config.def @llama.model.head_size = 64 : index
config.def @llama.model.kv_size = 512 : index
config.def @llama.model.layer_count = 1 : index
config.def @llama.model.rotary_step = {step!r} : f32
config.def @llama.attention.cache_capacity = 128 : index
config.def @llama.attention.pool_capacity = 128 : index
kernel.decl @llama32_rotary_query(%rows: index) launch(%kind: index, %origin: offset, %begin: index, %control: buffer, %factors: buffer, %input: buffer, %output: buffer)
kernel.decl @llama32_publish_kv(%rows: index) launch(%layer: index, %base: offset, %kind: index, %origin: offset, %begin: index, %map_origin: offset, %control: buffer, %map: buffer, %factors: buffer, %key: buffer, %value: buffer, %cache: buffer)
""",
        rodata("h64.factors", factors),
    ]
    for rows in [1, 8, 17]:
        for position in [0, 7, 63]:
            active = min(rows, 2)
            name = f"h64_r{rows}_p{position}"
            q = [f32(((i * 7) % 97 - 48) / 32) for i in range(rows * 2048)]
            k = [f32(((i * 11) % 83 - 41) / 32) for i in range(rows * 512)]
            v = [f32(((i * 13) % 71 - 35) / 16) for i in range(rows * 512)]
            expected_q = [-77.0] * (rows * 2048)
            expected_k = []
            expected_v = []
            for row in range(active):
                for source, heads, dest in [(q, 32, expected_q), (k, 8, expected_k)]:
                    for head in range(heads):
                        for pair in range(32):
                            theta = f32(position + row)
                            for _ in range(pair):
                                theta = f32(theta * step)
                            angle = f32(theta / factors[pair])
                            offset = row * heads * 64 + head * 64 + pair * 2
                            even, odd = source[offset : offset + 2]
                            a = f32(
                                f32(even * math.cos(angle)) - f32(odd * math.sin(angle))
                            )
                            b = f32(
                                f32(even * math.sin(angle)) + f32(odd * math.cos(angle))
                            )
                            if dest is expected_q:
                                dest[offset : offset + 2] = [a, b]
                            else:
                                dest.extend([a, b])
                expected_v.extend(v[row * 512 : (row + 1) * 512])
            for key, values in [
                ("q", q),
                ("k", k),
                ("v", v),
                ("expected_q", expected_q),
                ("expected_k", expected_k),
                ("expected_v", expected_v),
            ]:
                lines.append(rodata(f"{name}.{key}", values))
            lines.append(
                f"""func.def @{name}.init(%control: buffer, %map: buffer, %factors: buffer, %q: buffer, %k: buffer, %v: buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %control_origin = index.constant 4 : offset
  %map_origin = index.constant 8 : offset
  %ctl = buffer.view %control[%control_origin] : buffer -> view<3xi32>
  %mapping = buffer.view %map[%map_origin] : buffer -> view<2xi32>
  %active = scalar.constant {active} : i32
  %position = scalar.constant {position} : i32
  %zero32 = scalar.constant 0 : i32
  %one32 = scalar.constant 1 : i32
  view.store %active, %ctl[0] : i32, view<3xi32>
  view.store %position, %ctl[1] : i32, view<3xi32>
  view.store %one32, %mapping[0] : i32, view<2xi32>
  view.store %zero32, %mapping[1] : i32, view<2xi32>
"""
            )
            for key, count in [
                ("factors", 32),
                ("q", rows * 2048),
                ("k", rows * 512),
                ("v", rows * 512),
            ]:
                bank = "h64.factors" if key == "factors" else f"{name}.{key}"
                lines.append(f"""  %{key}_bank = global.load @{bank} : buffer
  %{key}_src = buffer.view %{key}_bank[%origin] : buffer -> view<{count}xf32>
  %{key}_dst = buffer.view %{key}[%origin] : buffer -> view<{count}xf32>
  %{key}_count = index.constant {count} : index
  scf.for %i = [%zero to %{key}_count step %one] {{
    %value = view.load %{key}_src[%i] : view<{count}xf32> -> f32
    view.store %value, %{key}_dst[%i] : f32, view<{count}xf32>
  }}""")
            lines.append(f"""  func.return
}}
command.def @{name}.actual() launch(%control: buffer, %map: buffer, %factors: buffer, %q: buffer, %k: buffer, %v: buffer, %output: buffer, %cache: buffer) {{
  %rows = index.constant {rows} : index
  %zero = index.constant 0 : index
  %origin = index.constant 4 : offset
  %map_origin = index.constant 8 : offset
  %base = index.constant 32 : offset
  kernel.launch @llama32_rotary_query[%rows](%zero, %origin, %zero, %control, %factors, %q, %output) : [index](index, offset, index, buffer, buffer, buffer, buffer)
  kernel.launch @llama32_publish_kv[%rows](%zero, %base, %zero, %origin, %zero, %map_origin, %control, %map, %factors, %k, %v, %cache) : [index](index, offset, index, offset, index, offset, buffer, buffer, buffer, buffer, buffer, buffer)
  command.return
}}
func.def @{name}.expected(%control: buffer, %map: buffer, %factors: buffer, %q: buffer, %k: buffer, %v: buffer, %output: buffer, %cache: buffer) {{
  %origin = index.constant 0 : offset
  %base = index.constant 32 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %qcount = index.constant {rows * 2048} : index
  %qbank = global.load @{name}.expected_q : buffer
  %qsrc = buffer.view %qbank[%origin] : buffer -> view<{rows * 2048}xf32>
  %qdst = buffer.view %output[%origin] : buffer -> view<{rows * 2048}xf32>
  scf.for %i = [%zero to %qcount step %one] {{
    %value = view.load %qsrc[%i] : view<{rows * 2048}xf32> -> f32
    view.store %value, %qdst[%i] : f32, view<{rows * 2048}xf32>
  }}
  %cache_view = buffer.view %cache[%base] : buffer -> view<2x128x512xf16>
""")
            for plane, key in [(0, "expected_k"), (1, "expected_v")]:
                lines.append(f"""  %{key}_bank = global.load @{name}.{key} : buffer
  %{key}_src = buffer.view %{key}_bank[%origin] : buffer -> view<{active}x512xf32>
""")
                for row in range(active):
                    logical = position + row
                    physical = (1 - logical // 64) * 64 + logical % 64
                    lines.append(f"""  %{key}_width_{row} = index.constant 512 : index
  scf.for %i = [%zero to %{key}_width_{row} step %one] {{
    %value = view.load %{key}_src[{row}, %i] : view<{active}x512xf32> -> f32
    %half = scalar.fptrunc %value : f32 to f16
    view.store %half, %cache_view[{plane}, {physical}, %i] : f16, view<2x128x512xf16>
  }}""")
            lines.append(f"""  func.return
}}
check.scenario @{name} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %control = check.generate.fill value(91) : tensor<8xi32>
    %map = check.generate.fill value(91) : tensor<8xi32>
    %factor_owner = check.generate.fill value(-99.0) : tensor<40xf32>
    %factors = check.tensor.view %factor_owner offset(16) : tensor<40xf32> -> tensor<32xf32>
    %q = check.generate.fill value(0.0) : tensor<{rows * 2048}xf32>
    %k = check.generate.fill value(0.0) : tensor<{rows * 512}xf32>
    %v = check.generate.fill value(0.0) : tensor<{rows * 512}xf32>
    %output_owner = check.generate.fill value(-77.0) : tensor<{rows * 2048 + 8}xf32>
    %output = check.tensor.view %output_owner offset(16) : tensor<{rows * 2048 + 8}xf32> -> tensor<{rows * 2048}xf32>
    %cache = check.generate.fill value(-77.0) : tensor<131104xf16>
    %cache_prefix = check.tensor.view %cache offset(0) : tensor<131104xf16> -> tensor<16xf16>
    %cache_suffix = check.tensor.view %cache offset(262176) : tensor<131104xf16> -> tensor<16xf16>
    %cache_untouched = check.tensor.view %cache offset(32800) : tensor<131104xf16> -> tensor<512xf16>
    %output_prefix = check.tensor.view %output_owner offset(0) : tensor<{rows * 2048 + 8}xf32> -> tensor<4xf32>
    %output_suffix = check.tensor.view %output_owner offset({4 * (rows * 2048 + 4)}) : tensor<{rows * 2048 + 8}xf32> -> tensor<4xf32>
    check.generate<@{name}.init>(%control, %map, %factors, %q, %k, %v) : (tensor<8xi32>, tensor<8xi32>, tensor<32xf32>, tensor<{rows * 2048}xf32>, tensor<{rows * 512}xf32>, tensor<{rows * 512}xf32>) -> ()
    check.compare<@{name}.actual, @{name}.expected>(%control, %map, %factors, %q, %k, %v, %output, %cache) : (tensor<8xi32>, tensor<8xi32>, tensor<32xf32>, tensor<{rows * 2048}xf32>, tensor<{rows * 512}xf32>, tensor<{rows * 512}xf32>, tensor<{rows * 2048}xf32>, tensor<131104xf16>) -> () {{
      check.expect.approximate actual(%output_owner) expected(%output_owner) atol(0.00001) rtol(0.0001) nan(same) : tensor<{rows * 2048 + 8}xf32>, tensor<{rows * 2048 + 8}xf32>
      check.expect.approximate actual(%cache) expected(%cache) atol(0.002) rtol(0.002) nan(same) : tensor<131104xf16>, tensor<131104xf16>
      check.expect.bitwise actual(%cache_prefix) expected(%cache_prefix) : tensor<16xf16>
      check.expect.bitwise actual(%cache_suffix) expected(%cache_suffix) : tensor<16xf16>
      check.expect.bitwise actual(%cache_untouched) expected(%cache_untouched) : tensor<512xf16>
      check.expect.bitwise actual(%output_prefix) expected(%output_prefix) : tensor<4xf32>
      check.expect.bitwise actual(%output_suffix) expected(%output_suffix) : tensor<4xf32>
      check.expect.bitwise actual(%factor_owner) expected(%factor_owner) : tensor<40xf32>
      check.expect.bitwise actual(%control) expected(%control) : tensor<8xi32>
      check.expect.bitwise actual(%map) expected(%map) : tensor<8xi32>
      check.expect.bitwise actual(%q) expected(%q) : tensor<{rows * 2048}xf32>
      check.expect.bitwise actual(%k) expected(%k) : tensor<{rows * 512}xf32>
      check.expect.bitwise actual(%v) expected(%v) : tensor<{rows * 512}xf32>
    }}
  }}
  check.return
}}""")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(emit())
