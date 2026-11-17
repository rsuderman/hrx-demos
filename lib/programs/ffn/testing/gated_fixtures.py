#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Canonical paired Q4 dots: independent logical anchors and exact GPU parity."""

import importlib.util
import sys
from pathlib import Path

helper, destination = sys.argv[1:]
spec = importlib.util.spec_from_file_location("fixture_helpers", helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.out.clear()
h.emit("// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n")
h.emit("config.def @ggml.linear_q4k_q8_1_x4.token_capacity = 512 : index\n")
h.emit("config.def @ggml.linear_q4k_q8_1_x4.output_capacity = 17408 : index\n")
h.emit("config.def @ggml.quantize_q8_1_x4.group_capacity = 16384 : index\n")
h.emit(
    """kernel.decl @ggml_quantize_q8_1_x4_f32(%R:index,%K:index) launch(%C:index,%K:index,%input:buffer,%packed:buffer)
kernel.decl @ggml_linear_q4k_q8_1_x4(%R:index,%K:index,%N:index) launch(%C:index,%K:index,%N:index,%packed:buffer,%weight:buffer,%output:buffer)
kernel.decl @llm.projection.q4k_swiglu(%R:index,%K:index,%N:index) launch(%C:index,%K:index,%N:index,%packed:buffer,%gate:buffer,%up:buffer,%output:buffer)
kernel.decl @llm_ffn_swiglu_f32_prefill(%R:index,%N:index) launch(%N:index,%C:index,%gate:buffer,%up:buffer,%output:buffer)
"""
)

# Capacity may exceed the launched rows. The untouched tail and both guards
# remain sentinels; non-power-of-two K exercises the dot helper's block tail.
for case, (R, C, K, N) in enumerate(
    (
        (1, 1, 256, 256),
        (1, 4, 768, 256),
        (4, 4, 1024, 256),
        (16, 16, 2560, 256),
        (1, 512, 1024, 3584),
        (1, 4, 5120, 17408),
        (1, 1, 256, 512),
        (1, 1, 256, 13),
    )
):
    name = f"paired_q4_r{R}_c{C}_k{K}_n{N}"
    W, P, O = K * N // 256 * 144, C * K // 128 * 144, C * N
    args = "%input:buffer,%gate:buffer,%up:buffer,%output:buffer"
    tensor_types = f"tensor<{C*K}xf32>,tensor<{W}xi8>,tensor<{W}xi8>,tensor<{O}xf32>"
    body = h.prefix() + h.input_body(R, K, True).replace(
        "0.9921875", "0.0" if N == 512 else "0.9921875"
    )
    body += h.weight_body(12, K, "gate", "gate", "g")
    body += h.weight_body(12, K, "up", "up", "u")
    h.kernel(name + ".init", N, "%input:buffer,%gate:buffer,%up:buffer", body)
    h.reference(
        name + ".init",
        N,
        "%input:buffer,%gate:buffer,%up:buffer",
        f"    func.call @{name}.init.body(%channel,%input,%gate,%up) : (index,buffer,buffer,buffer) -> ()",
    )
    _, anchors, _ = h.ffn_anchors(R, K, N)
    if N == 512:
        anchors = [[0.0] * 4 for _ in range(R)]
    h.kernel(
        "test.expected." + name,
        N,
        "%output:buffer",
        h.prefix() + h.store_pattern(R, N, "output", anchors, "out"),
    )
    h.reference(
        name,
        N,
        args,
        f"    func.call @test.expected.{name}.body(%channel,%output) : (index,buffer) -> ()",
    )
    constants = f"""  %R = index.constant {R} : index
  %C = index.constant {C} : index
  %K = index.constant {K} : index
  %N = index.constant {N} : index
  %pb = index.constant {P} : offset
  %ob = index.constant {O*4} : offset
  %packed = buffer.alloca<global> align(256) %pb : buffer
  kernel.launch @ggml_quantize_q8_1_x4_f32[%R,%K](%C,%K,%input,%packed) : [index,index](index,index,buffer,buffer)
"""
    fused = "  kernel.launch @llm.projection.q4k_swiglu[%R,%K,%N](%C,%K,%N,%packed,%gate,%up,%output) : [index,index,index](index,index,index,buffer,buffer,buffer,buffer)\n"
    separate = """  %g = buffer.alloca<global> align(256) %ob : buffer
  %u = buffer.alloca<global> align(256) %ob : buffer
  kernel.launch @ggml_linear_q4k_q8_1_x4[%R,%K,%N](%C,%K,%N,%packed,%gate,%g) : [index,index,index](index,index,index,buffer,buffer,buffer)
  kernel.launch @ggml_linear_q4k_q8_1_x4[%R,%K,%N](%C,%K,%N,%packed,%up,%u) : [index,index,index](index,index,index,buffer,buffer,buffer)
  kernel.launch @llm_ffn_swiglu_f32_prefill[%R,%N](%N,%C,%g,%u,%output) : [index,index](index,index,buffer,buffer,buffer)
"""
    routes = ["fused"] if N == 13 else ["fused", "baseline"]
    for route in routes:
        h.emit(
            f"command.def @{name}.{route}() launch({args}) {{\n"
            + constants
            + (fused if route == "fused" else separate)
            + "  command.return\n}\n"
        )
        h.emit(f"""check.scenario @{name}_{route} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input_owner = check.generate.fill value(-77.0) : tensor<{C*K+8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{C*K+8}xf32> -> tensor<{C*K}xf32>
    %gate_owner = check.generate.fill value(0) : tensor<{W+512}xi8>
    %gate = check.tensor.view %gate_owner offset(256) : tensor<{W+512}xi8> -> tensor<{W}xi8>
    %up_owner = check.generate.fill value(0) : tensor<{W+512}xi8>
    %up = check.tensor.view %up_owner offset(256) : tensor<{W+512}xi8> -> tensor<{W}xi8>
    %owner = check.generate.fill value(-99.0) : tensor<{O+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{O+8}xf32> -> tensor<{O}xf32>
    check.generate<@reference.{name}.init>(%input,%gate,%up) : (tensor<{C*K}xf32>,tensor<{W}xi8>,tensor<{W}xi8>) -> ()
    check.compare<@{name}.{route},@reference.{name}>(%input,%gate,%up,%output) : ({tensor_types}) -> () {{
      check.expect.approximate actual(%owner) expected(%owner) atol(0.0001) rtol(0.0001) nan(same) : tensor<{O+8}xf32>,tensor<{O+8}xf32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{C*K+8}xf32>
      check.expect.bitwise actual(%gate_owner) expected(%gate_owner) : tensor<{W+512}xi8>
      check.expect.bitwise actual(%up_owner) expected(%up_owner) : tensor<{W+512}xi8>
    }}
  }}
  check.return
}}
check.benchmark<@{name}_{route}> @bench_{name}_{route}
""")
    if N == 13:
        continue
    # Both numerical producers run on the GPU. A separate integer-only checker
    # compares every activated F32 bit; the VM oracle requires zero mismatches.
    h.kernel(
        name + ".compare",
        N,
        "%lhs:buffer,%rhs:buffer,%flags:buffer",
        h.prefix()
        + f"""  %lhs_v = buffer.view %lhs[%origin] : buffer -> view<{R}x{N}xi32>
  %rhs_v = buffer.view %rhs[%origin] : buffer -> view<{R}x{N}xi32>
  %flags_v = buffer.view %flags[%origin] : buffer -> view<{R}x{N}xi32>
  %end = index.constant {R} : index
  scf.for %r = [%zero to %end step %one] {{
    %a = view.load %lhs_v[%r,%channel] : view<{R}x{N}xi32> -> i32
    %b = view.load %rhs_v[%r,%channel] : view<{R}x{N}xi32> -> i32
    %different = scalar.cmpi ne, %a, %b : i32
    %flag = scalar.extui %different : i1 to i32
    view.store %flag, %flags_v[%r,%channel] : i32, view<{R}x{N}xi32>
  }}""",
    )
    h.emit(f"""command.def @{name}.parity() launch({args},%flags:buffer) {{
{constants}  %baseline = buffer.alloca<global> align(256) %ob : buffer
{fused}{separate.replace("%output", "%baseline")}  kernel.launch @{name}.compare(%output,%baseline,%flags) : (buffer,buffer,buffer)
  command.return
}}
func.def @{name}.parity_expected({args},%flags:buffer) {{
  %o = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %end = index.constant {R*N} : index
  %z = scalar.constant 0 : i32
  %f = buffer.view %flags[%o] : buffer -> view<{R*N}xi32>
  scf.for %i = [%zero to %end step %one] {{
    view.store %z,%f[%i] : i32,view<{R*N}xi32>
  }}
  func.return
}}
check.scenario @{name}_parity {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-77.0) : tensor<{C*K}xf32>
    %gate = check.generate.fill value(0) : tensor<{W}xi8>
    %up = check.generate.fill value(0) : tensor<{W}xi8>
    %output = check.generate.fill value(-99.0) : tensor<{O}xf32>
    %flags = check.generate.fill value(-1) : tensor<{R*N}xi32>
    check.generate<@reference.{name}.init>(%input,%gate,%up) : (tensor<{C*K}xf32>,tensor<{W}xi8>,tensor<{W}xi8>) -> ()
    check.compare<@{name}.parity,@{name}.parity_expected>(%input,%gate,%up,%output,%flags) : ({tensor_types},tensor<{R*N}xi32>) -> () {{
      check.expect.bitwise actual(%flags) expected(%flags) : tensor<{R*N}xi32>
    }}
  }}
  check.return
}}
""")
Path(destination).write_text("".join(h.out))
