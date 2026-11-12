#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Exact power-of-two scaling and active-prefix padding witnesses."""

from pathlib import Path
import sys


def emit():
    text = ["// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n"]
    text.append(
        "kernel.decl @llm.projection.prepare_scaled_f16(%T:index,%Tp:index,%K:index) launch(%T:index,%Tp:index,%K:index,%origin:offset,%count:buffer,%input:buffer,%scales:buffer,%output:buffer)\n"
    )
    for t, active in [
        (17, 1),
        (17, 17),
        (64, 63),
        (128, 65),
        (128, 1),
        (256, 129),
        (512, 257),
        (512, 511),
    ]:
        tp = (t + 63) // 64 * 64
        name = f"prepare_t{t}_n{active}"
        text.append(f"""func.def @{name}.init(%input:buffer,%count:buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %five = index.constant 5 : index
  %K = index.constant 256 : index
  %N = index.constant {active} : index
  %source = buffer.view %input[%origin] : buffer -> view<{t}x256xf32>
  %counts = buffer.view %count[%origin] : buffer -> view<3xi32>
  %n = scalar.constant {active} : i32
  view.store %n, %counts[0] : i32, view<3xi32>
  %a = scalar.constant 32768.0 : f32
  %b = scalar.constant 16384.0 : f32
  %bias = scalar.constant 2 : i32
  scf.for %row = [%zero to %N step %one] {{
    %parity = index.rem %row, %two : index
    %even = index.cmp eq, %parity, %zero : index
    %factor = scf.select %even, %a, %b : f32
    scf.for %column = [%zero to %K step %one] {{
      %phase = index.rem %column, %five : index
      %phase32 = index.cast %phase : index to i32
      %signed = scalar.subi %phase32, %bias : i32
      %integer = scalar.sitofp %signed : i32 to f32
      %value = scalar.mulf %integer, %factor : f32
      view.store %value, %source[%row,%column] : f32, view<{t}x256xf32>
    }}
  }}
  func.return
}}
command.def @{name}.actual() launch(%input:buffer,%count:buffer,%scales:buffer,%output:buffer) {{
  %T = index.constant {t} : index
  %Tp = index.constant {tp} : index
  %K = index.constant 256 : index
  %origin = index.constant 0 : offset
  kernel.launch @llm.projection.prepare_scaled_f16[%T,%Tp,%K](%T,%Tp,%K,%origin,%count,%input,%scales,%output) : [index,index,index](index,index,index,offset,buffer,buffer,buffer,buffer)
  command.return
}}
func.def @{name}.expected(%input:buffer,%count:buffer,%scales:buffer,%output:buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %five = index.constant 5 : index
  %K = index.constant 256 : index
  %N = index.constant {active} : index
  %Tp = index.constant {tp} : index
  %metadata = buffer.view %scales[%origin] : buffer -> view<{tp}x2xf32>
  %destination = buffer.view %output[%origin] : buffer -> view<{tp}x256xf16>
  %scale0 = scalar.constant 0.125 : f32
  %scale1 = scalar.constant 0.25 : f32
  %inverse0 = scalar.constant 8.0 : f32
  %inverse1 = scalar.constant 4.0 : f32
  %z = scalar.constant 0.0 : f32
  %factor = scalar.constant 4096.0 : f32
  %bias = scalar.constant 2 : i32
  scf.for %row = [%zero to %Tp step %one] {{
    %live = index.cmp ult, %row, %N : index
    %parity = index.rem %row, %two : index
    %even = index.cmp eq, %parity, %zero : index
    %s = scf.select %even, %scale0, %scale1 : f32
    %i = scf.select %even, %inverse0, %inverse1 : f32
    %scale = scf.select %live, %s, %z : f32
    %inverse = scf.select %live, %i, %z : f32
    view.store %scale, %metadata[%row,0] : f32, view<{tp}x2xf32>
    view.store %inverse, %metadata[%row,1] : f32, view<{tp}x2xf32>
    scf.for %column = [%zero to %K step %one] {{
      %phase = index.rem %column, %five : index
      %phase32 = index.cast %phase : index to i32
      %signed = scalar.subi %phase32, %bias : i32
      %integer = scalar.sitofp %signed : i32 to f32
      %value0 = scalar.mulf %integer, %factor : f32
      %value = scf.select %live, %value0, %z : f32
      %half = scalar.fptrunc %value : f32 to f16
      view.store %half, %destination[%row,%column] : f16, view<{tp}x256xf16>
    }}
  }}
  func.return
}}
check.scenario @{name} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-777.0) : tensor<{t*256}xf32>
    %count = check.generate.fill value(0) : tensor<3xi32>
    %scale_owner = check.generate.fill value(-991.0) : tensor<{tp*2+8}xf32>
    %scales = check.tensor.view %scale_owner offset(16) : tensor<{tp*2+8}xf32> -> tensor<{tp*2}xf32>
    %owner = check.generate.fill value(-991.0) : tensor<{tp*256+16}xf16>
    %output = check.tensor.view %owner offset(16) : tensor<{tp*256+16}xf16> -> tensor<{tp*256}xf16>
    check.generate<@{name}.init>(%input,%count) : (tensor<{t*256}xf32>,tensor<3xi32>) -> ()
    check.compare<@{name}.actual,@{name}.expected>(%input,%count,%scales,%output) : (tensor<{t*256}xf32>,tensor<3xi32>,tensor<{tp*2}xf32>,tensor<{tp*256}xf16>) -> () {{
      check.expect.bitwise actual(%scale_owner) expected(%scale_owner) : tensor<{tp*2+8}xf32>
      check.expect.bitwise actual(%owner) expected(%owner) : tensor<{tp*256+16}xf16>
      check.expect.bitwise actual(%input) expected(%input) : tensor<{t*256}xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
    }}
  }}
  check.return
}}
""")
    return "".join(text)


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(emit())
