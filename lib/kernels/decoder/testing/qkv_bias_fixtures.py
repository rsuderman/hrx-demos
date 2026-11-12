#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent F64 bias/NeoX anchors with explicit F32 wire rounding.

No model execution or production numerical helper is used for expectations.
Non-unit head amplitudes distinguish missing QK normalization from gamma=1 RMS.
"""

import math
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def emit():
    lines = ["// Generated independent bias/NeoX fixtures."]
    lines += [
        "kernel.decl @llm.decoder.prepare_qk_bias_h128(%rows: index, %heads: index) launch(%capacity: index, %step: f32, %origin: offset, %epoch: buffer, %input: buffer, %bias: buffer, %output: buffer)",
        "kernel.decl @llm.decoder.add_bias_active(%rows: index, %width: index) launch(%width: index, %count: buffer, %bias: buffer, %projection: buffer)",
    ]
    cases = [
        ("q0", 12, 0, 2),
        ("q7", 12, 7, 2),
        ("k63", 2, 63, 2),
        ("zero", 2, 7, 0),
        ("negative_count", 2, 7, -1),
        ("invalid_count", 2, 7, 4),
        ("capacity_end", 2, 127, 2),
        ("past_end", 2, 128, 2),
        ("negative_position", 2, -1, 2),
    ]
    step = f32(-2 * math.log(1000000.0) / 128)
    for name, heads, base, active in cases:
        width, size = heads * 128, 3 * heads * 128
        inputs = [
            (
                f32(2 + (i % 43) / 16)
                if i < min(active, 3) * width and 0 < active <= 3
                else math.nan
            )
            for i in range(size)
        ]
        bias = [f32((i % 19 - 9) / 32) for i in range(width)]
        expected = [0.0] * size
        for row in range(3):
            if not (
                0 < active <= 3 and row < active and base >= 0 and base + row < 128
            ):
                continue
            for head in range(heads):
                for lane in range(128):
                    pair = lane % 64
                    lo = f32(
                        inputs[row * width + head * 128 + pair]
                        + bias[head * 128 + pair]
                    )
                    hi = f32(
                        inputs[row * width + head * 128 + pair + 64]
                        + bias[head * 128 + pair + 64]
                    )
                    angle = f32((base + row) * f32(math.exp(f32(pair * step))))
                    value = (
                        lo * math.cos(angle) - hi * math.sin(angle)
                        if lane < 64
                        else lo * math.sin(angle) + hi * math.cos(angle)
                    )
                    expected[row * width + head * 128 + lane] = f32(value)
        for key, values in [("input", inputs), ("bias", bias), ("expected", expected)]:
            blob = b"".join(struct.pack("<f", v) for v in values).hex()
            lines.append(
                f'global.rodata.def @bias.oracle.{name}.{key} = bytes("{blob}")'
            )
        first_view = (
            f"    %first = check.tensor.view %out offset(0) : tensor<{size}xf32> -> tensor<{width}xf32>\n"
            if name == "q0"
            else ""
        )
        first_assert = (
            f"      check.expect.bitwise actual(%first) expected(%first) : tensor<{width}xf32>\n"
            if name == "q0"
            else ""
        )
        lines.append(
            f"""func.def @{name}.init(%epoch: buffer, %input: buffer, %bias: buffer) {{
  %origin = index.constant 0 : offset
  %descriptor = index.constant 12 : offset
  %epoch_view = buffer.view %epoch[%descriptor] : buffer -> view<5xi32>
  %n = scalar.constant {active} : i32
  %base = scalar.constant {base} : i32
  %zero32 = scalar.constant 0 : i32
  view.store %n, %epoch_view[0] : i32, view<5xi32>
  view.store %base, %epoch_view[1] : i32, view<5xi32>
  view.store %zero32, %epoch_view[2] : i32, view<5xi32>
  view.store %zero32, %epoch_view[3] : i32, view<5xi32>
  view.store %zero32, %epoch_view[4] : i32, view<5xi32>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
"""
        )
        for key, count in [("input", size), ("bias", width)]:
            lines.append(
                f"""  %{key}_bank = global.load @bias.oracle.{name}.{key} : buffer
  %{key}_src = buffer.view %{key}_bank[%origin] : buffer -> view<{count}xf32>
  %{key}_dst = buffer.view %{key}[%origin] : buffer -> view<{count}xf32>
  %{key}_size = index.constant {count} : index
  scf.for %i = [%zero to %{key}_size step %one] {{
    %v = view.load %{key}_src[%i] : view<{count}xf32> -> f32
    view.store %v, %{key}_dst[%i] : f32, view<{count}xf32>
  }}"""
            )
        lines.append(f"""  func.return
}}
command.def @{name}.actual() launch(%epoch: buffer, %input: buffer, %bias: buffer, %output: buffer) {{
  %rows = index.constant 3 : index
  %heads = index.constant {heads} : index
  %capacity = index.constant 128 : index
  %step = scalar.constant {step!r} : f32
  %origin = index.constant 12 : offset
  kernel.launch @llm.decoder.prepare_qk_bias_h128[%rows, %heads](%capacity, %step, %origin, %epoch, %input, %bias, %output) : [index, index](index, f32, offset, buffer, buffer, buffer, buffer)
  command.return
}}
func.def @{name}.expected(%epoch: buffer, %input: buffer, %bias: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %size = index.constant {size} : index
  %bank = global.load @bias.oracle.{name}.expected : buffer
  %source = buffer.view %bank[%origin] : buffer -> view<{size}xf32>
  %dest = buffer.view %output[%origin] : buffer -> view<{size}xf32>
  scf.for %i = [%zero to %size step %one] {{
    %v = view.load %source[%i] : view<{size}xf32> -> f32
    view.store %v, %dest[%i] : f32, view<{size}xf32>
  }}
  func.return
}}
check.scenario @qkv_bias.{name} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %epoch = check.generate.fill value(99) : tensor<12xi32>
    %input = check.generate.fill value(99.0) : tensor<{size}xf32>
    %bias = check.generate.fill value(99.0) : tensor<{width}xf32>
    %owner = check.generate.fill value(-77.0) : tensor<{size+8}xf32>
    %out = check.tensor.view %owner offset(16) : tensor<{size+8}xf32> -> tensor<{size}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{size+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({4*(size+4)}) : tensor<{size+8}xf32> -> tensor<4xf32>
    %inactive = check.tensor.view %out offset({8*width}) : tensor<{size}xf32> -> tensor<{width}xf32>
{first_view}    check.generate<@{name}.init>(%epoch, %input, %bias) : (tensor<12xi32>, tensor<{size}xf32>, tensor<{width}xf32>) -> ()
    check.compare<@{name}.actual, @{name}.expected>(%epoch, %input, %bias, %out) : (tensor<12xi32>, tensor<{size}xf32>, tensor<{width}xf32>, tensor<{size}xf32>) -> () {{
      check.expect.approximate actual(%out) expected(%out) atol(0.00001) rtol(0.0001) nan(same) : tensor<{size}xf32>, tensor<{size}xf32>
{first_assert}      check.expect.bitwise actual(%inactive) expected(%inactive) : tensor<{width}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>
      check.expect.bitwise actual(%epoch) expected(%epoch) : tensor<12xi32>
      check.expect.bitwise actual(%input) expected(%input) : tensor<{size}xf32>
      check.expect.bitwise actual(%bias) expected(%bias) : tensor<{width}xf32>
    }}
  }}
  check.return
}}""")
    # Exact elementwise bias anchors, including all inactive rows and guards.
    for active in [-1, 0, 1, 2, 4]:
        name = f"v{active}" if active >= 0 else "v_negative"
        v_bias = [f32((i % 19 - 9) / 32) for i in range(256)]
        v_input = [f32(2 + (i % 43) / 16) for i in range(768)]
        v_expected = [
            f32(v + v_bias[i % 256]) if 0 <= i < active * 256 and active <= 3 else v
            for i, v in enumerate(v_input)
        ]
        for key, values in [
            ("input", v_input),
            ("bias", v_bias),
            ("expected", v_expected),
        ]:
            blob = b"".join(struct.pack("<f", v) for v in values).hex()
            lines.append(
                f'global.rodata.def @bias.oracle.{name}.{key} = bytes("{blob}")'
            )
        lines.append(f"""func.def @{name}.init(%bias: buffer, %projection: buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
""")
        for key, argument, count in [
            ("input", "projection", 768),
            ("bias", "bias", 256),
        ]:
            lines.append(
                f"""  %{key}_bank = global.load @bias.oracle.{name}.{key} : buffer
  %{key}_src = buffer.view %{key}_bank[%origin] : buffer -> view<{count}xf32>
  %{key}_dst = buffer.view %{argument}[%origin] : buffer -> view<{count}xf32>
  %{key}_size = index.constant {count} : index
  scf.for %i = [%zero to %{key}_size step %one] {{
    %v = view.load %{key}_src[%i] : view<{count}xf32> -> f32
    view.store %v, %{key}_dst[%i] : f32, view<{count}xf32>
  }}"""
            )
        lines.append("  func.return\n}")
        lines.append(
            f"""command.def @{name}.actual() launch(%count: buffer, %bias: buffer, %projection: buffer) {{
  %rows = index.constant 3 : index
  %width = index.constant 256 : index
  kernel.launch @llm.decoder.add_bias_active[%rows, %width](%width, %count, %bias, %projection) : [index, index](index, buffer, buffer, buffer)
  command.return
}}
func.def @{name}.expected(%count: buffer, %bias: buffer, %projection: buffer) {{
  %origin = index.constant 0 : offset
  %out = buffer.view %projection[%origin] : buffer -> view<768xf32>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %size = index.constant 768 : index
  %bank = global.load @bias.oracle.{name}.expected : buffer
  %source = buffer.view %bank[%origin] : buffer -> view<768xf32>
  scf.for %i = [%zero to %size step %one] {{
    %value = view.load %source[%i] : view<768xf32> -> f32
    view.store %value, %out[%i] : f32, view<768xf32>
  }}
  func.return
}}
check.scenario @qkv_bias.{name} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %bias = check.generate.fill value(0.75) : tensor<256xf32>
    %owner = check.generate.fill value(2.5) : tensor<776xf32>
    %out = check.tensor.view %owner offset(16) : tensor<776xf32> -> tensor<768xf32>
    check.generate<@{name}.init>(%bias, %out) : (tensor<256xf32>, tensor<768xf32>) -> ()
    check.compare<@{name}.actual, @{name}.expected>(%count, %bias, %out) : (tensor<3xi32>, tensor<256xf32>, tensor<768xf32>) -> () {{
      check.expect.bitwise actual(%owner) expected(%owner) : tensor<776xf32>
      check.expect.bitwise actual(%bias) expected(%bias) : tensor<256xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
    }}
  }}
  check.return
}}"""
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(emit())
