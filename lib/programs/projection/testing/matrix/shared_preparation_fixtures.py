#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Exact nonzero sibling projection anchors and matched preparation benchmarks.

Uniform +/-127*factor inputs quantize exactly (Q8 scale=factor). Independently
encoded Q4_K weights are 1.5/-0.75 and Q6_K weights are 0.25, so all row dots
are exact F32 integers under both canonical-Q8 and scaled-F16 policies. No
production decoder or saved model activations enter the host oracle.
"""

import sys
from pathlib import Path

QKV_ARGS = "%T:index,%K:index,%Q:index,%KV:index,%qt:index,%kt:index,%vt:index,%qb:index,%kb:index,%vb:index,%count:buffer,%input:buffer,%qw:view<[%qb]xi8>,%kw:view<[%kb]xi8>,%vw:view<[%vb]xi8>,%q:buffer,%k:buffer,%v:buffer"


def emit():
    out = ["// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n"]
    out.append(f"template.decl @llm.projection.matrix_qkv({QKV_ARGS})\n")
    out.append(
        "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%weight:view<[%bytes]xi8>,%output:buffer)\n"
        "template.decl @llm.projection.prepare_canonical_prefix(%T:index,%K:index,%count:buffer,%input:buffer,%packed:buffer)\n"
    )
    for tag in ("q4k", "q5_0", "q5k", "q6k", "q8_0"):
        out.append(
            f"config.def @ggml.linear_{tag}_q8_1_x4.token_capacity = 512 : index\n"
            f"config.def @ggml.linear_{tag}_q8_1_x4.output_capacity = 256 : index\n"
        )
    out.append("config.def @ggml.quantize_q8_1_x4.group_capacity = 1024 : index\n")
    cases = (
        (1, 0),
        (1, 1),
        (4, 1),
        (4, 4),
        (16, 0),
        (16, 15),
        (16, 16),
        (17, 1),
        (17, 17),
        (64, 63),
        (64, 64),
        (65, 64),
        (65, 65),
        (128, 65),
        (512, 511),
        (512, 512),
    )
    for t, active in cases:
        name = f"shared_qkv_t{t}_a{active}"
        padded = t if t <= 16 else (t + 63) // 64 * 64
        qb, kb, vb = 256 * 144, 128 * 210, 128 * 144
        out.append(
            f"""func.def @{name}.init(%input:buffer,%count:buffer,%qw:buffer,%kw:buffer,%vw:buffer) {{
  %o = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %T = index.constant {active} : index
  %K = index.constant 256 : index
  %Q = index.constant 256 : index
  %KV = index.constant 128 : index
  %a = scalar.constant {active} : i32
  %count_view = buffer.view %count[%o] : buffer -> view<3xi32>
  view.store %a, %count_view[0] : i32, view<3xi32>
  %x = buffer.view %input[%o] : buffer -> view<{t}x256xf32>
  scf.for %row = [%zero to %T step %one] {{
    %phase = index.rem %row, %three : index
    %factor = index.add %phase, %one : index
    %f32 = index.cast %factor : index to i32
    %f = scalar.sitofp %f32 : i32 to f32
    %parity = index.rem %row, %two : index
    %even = index.cmp eq, %parity, %zero : index
    %positive = scalar.constant 127.0 : f32
    %negative = scalar.constant -127.0 : f32
    %unit = scf.select %even, %positive, %negative : f32
    %value = scalar.mulf %unit, %f : f32
    scf.for %column = [%zero to %K step %one] {{
      view.store %value, %x[%row,%column] : f32, view<{t}x256xf32>
    }}
  }}
"""
        )
        for role, width, tag, d in (
            ("q", 256, 12, 0.5),
            ("k", 128, 14, 0.25),
            ("v", 128, 12, -0.25),
        ):
            block_bytes = 144 if tag == 12 else 210
            out.append(f"""  %rows_{role} = index.constant {width} : index
  %stride_{role} = index.constant {block_bytes} : offset
  scf.for %channel = [%zero to %rows_{role} step %one] {{
    %base = index.scale %channel, %stride_{role} : index, offset -> offset
    %block_{role} = buffer.view %{role}w[%base] : buffer -> view<{block_bytes}xi8>
""")
            # Encode the wire representation explicitly, independently of kernels.
            if tag == 12:
                out.append(
                    f"""    %header = buffer.view %{role}w[%base] : buffer -> view<2xf16>
    %d = scalar.constant {d} : f16
    %z = scalar.constant 0.0 : f16
    view.store %d, %header[0] : f16, view<2xf16>
    view.store %z, %header[1] : f16, view<2xf16>
    %sc_begin = index.constant 4 : index
    %sc_end = index.constant 16 : index
    %end = index.constant 144 : index
    %scale = scalar.constant 1 : i8
    %codes = scalar.constant 51 : i8
    scf.for %byte = [%sc_begin to %sc_end step %one] {{
      view.store %scale, %block_{role}[%byte] : i8, view<144xi8>
    }}
    scf.for %byte = [%sc_end to %end step %one] {{
      view.store %codes, %block_{role}[%byte] : i8, view<144xi8>
    }}
"""
                )
            else:
                out.append(f"""    %low_end = index.constant 128 : index
    %high_end = index.constant 192 : index
    %scale_end = index.constant 208 : index
    %low = scalar.constant 17 : i8
    %high = scalar.constant -86 : i8
    %scale = scalar.constant 1 : i8
    scf.for %byte = [%zero to %low_end step %one] {{
      view.store %low, %block_{role}[%byte] : i8, view<210xi8>
    }}
    scf.for %byte = [%low_end to %high_end step %one] {{
      view.store %high, %block_{role}[%byte] : i8, view<210xi8>
    }}
    scf.for %byte = [%high_end to %scale_end step %one] {{
      view.store %scale, %block_{role}[%byte] : i8, view<210xi8>
    }}
    %header_add = index.constant 208 : offset
    %header_base = index.add %base, %header_add : offset
    %header = buffer.view %{role}w[%header_base] : buffer -> view<1xf16>
    %d = scalar.constant {d} : f16
    view.store %d, %header[0] : f16, view<1xf16>
""")
            out.append("  }\n")
        out.append("  func.return\n}\n")
        args = "%input:buffer,%count:buffer,%qw:buffer,%kw:buffer,%vw:buffer,%q:buffer,%k:buffer,%v:buffer"
        constants = f"""  %T = index.constant {t} : index
  %K = index.constant 256 : index
  %Q = index.constant 256 : index
  %KV = index.constant 128 : index
  %qtag = index.constant 12 : index
  %ktag = index.constant 14 : index
  %qb = index.constant {qb} : index
  %kb = index.constant {kb} : index
  %vb = index.constant {vb} : index
  %origin = index.constant 0 : offset
  %qweight = buffer.view %qw[%origin] : buffer -> view<[%qb]xi8>
  %kweight = buffer.view %kw[%origin] : buffer -> view<[%kb]xi8>
  %vweight = buffer.view %vw[%origin] : buffer -> view<[%vb]xi8>
"""
        for route in ("candidate", "baseline"):
            out.append(f"command.def @{name}.{route}() launch({args}) {{\n" + constants)
            if route == "candidate":
                out.append(
                    f"  template.apply<@llm.projection.matrix_qkv>(%T,%K,%Q,%KV,%qtag,%ktag,%qtag,%qb,%kb,%vb,%count,%input,%qweight,%kweight,%vweight,%q,%k,%v) : (index,index,index,index,index,index,index,index,index,index,buffer,buffer,view<[%qb]xi8>,view<[%kb]xi8>,view<[%vb]xi8>,buffer,buffer,buffer)\n"
                )
            else:
                for role, width, tag, wb in (
                    ("q", "Q", "qtag", qb),
                    ("k", "KV", "ktag", kb),
                    ("v", "KV", "qtag", vb),
                ):
                    out.append(
                        f"  template.apply<@llm.projection.matrix>(%T,%K,%{width},%{tag},%{role}b,%count,%input,%{role}weight,%{role}) : (index,index,index,index,index,buffer,buffer,view<[%{role}b]xi8>,buffer)\n"
                    )
            out.append("  command.return\n}\n")
        out.append(f"""func.def @{name}.expected({args}) {{
  %o = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %P = index.constant {padded} : index
  %A = index.constant {active} : index
""")
        for role, width, dot in (
            ("q", 256, 48768.0),
            ("k", 128, 8128.0),
            ("v", 128, -24384.0),
        ):
            out.append(f"""  %N_{role} = index.constant {width} : index
  %out_{role} = buffer.view %{role}[%o] : buffer -> view<{padded}x{width}xf32>
  %unit_{role} = scalar.constant {dot} : f32
  scf.for %row = [%zero to %P step %one] {{
    %live = index.cmp ult, %row, %A : index
    %phase = index.rem %row, %three : index
    %factor = index.add %phase, %one : index
    %factor32 = index.cast %factor : index to i32
    %f = scalar.sitofp %factor32 : i32 to f32
    %parity = index.rem %row, %two : index
    %even = index.cmp eq, %parity, %zero : index
    %pos = scalar.constant 1.0 : f32
    %neg = scalar.constant -1.0 : f32
    %sign = scf.select %even, %pos, %neg : f32
    %signed = scalar.mulf %f, %sign : f32
    %dot = scalar.mulf %signed, %unit_{role} : f32
    %z = scalar.constant 0.0 : f32
    %value = scf.select %live, %dot, %z : f32
    scf.for %column = [%zero to %N_{role} step %one] {{
      view.store %value, %out_{role}[%row,%column] : f32, view<{padded}x{width}xf32>
    }}
  }}
""")
        out.append("  func.return\n}\n")
        tensor_types = (
            f"tensor<{t*256}xf32>",
            "tensor<3xi32>",
            f"tensor<{qb}xi8>",
            f"tensor<{kb}xi8>",
            f"tensor<{vb}xi8>",
            f"tensor<{padded*256}xf32>",
            f"tensor<{padded*128}xf32>",
            f"tensor<{padded*128}xf32>",
        )
        for route in ("candidate", "baseline"):
            scenario = f"{name}_{route}"
            out.append(f"""check.scenario @{scenario} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-777.0) : tensor<{t*256}xf32>
    %count = check.generate.fill value(0) : tensor<3xi32>
    %qw = check.generate.fill value(0) : tensor<{qb}xi8>
    %kw = check.generate.fill value(0) : tensor<{kb}xi8>
    %vw = check.generate.fill value(0) : tensor<{vb}xi8>
""")
            for role, width in (("q", 256), ("k", 128), ("v", 128)):
                n = padded * width
                out.append(
                    f"    %{role}_owner = check.generate.fill value(-991.0) : tensor<{n+8}xf32>\n    %{role} = check.tensor.view %{role}_owner offset(16) : tensor<{n+8}xf32> -> tensor<{n}xf32>\n"
                )
            out.append(
                f"    check.generate<@{name}.init>(%input,%count,%qw,%kw,%vw) : ({','.join(tensor_types[:5])}) -> ()\n"
            )
            out.append(
                f"    check.compare<@{name}.{route},@{name}.expected>(%input,%count,%qw,%kw,%vw,%q,%k,%v) : ({','.join(tensor_types)}) -> () {{\n"
            )
            for role, width in (("q", 256), ("k", 128), ("v", 128)):
                out.append(
                    f"      check.expect.bitwise actual(%{role}_owner) expected(%{role}_owner) : tensor<{padded*width+8}xf32>\n"
                )
            for role, ty in zip(("input", "count", "qw", "kw", "vw"), tensor_types):
                out.append(
                    f"      check.expect.bitwise actual(%{role}) expected(%{role}) : {ty}\n"
                )
            out.append("    }\n  }\n  check.return\n}\n")
            out.append(f"check.benchmark<@{scenario}> @bench_{scenario}\n")
    return "".join(out) + prefix_cases() + gated_cases()


def prefix_cases():
    out = []
    for t, active in ((1, 0), (1, 1), (4, 1), (16, 0), (16, 15), (16, 16)):
        name = f"shared_prefix_t{t}_a{active}"
        packed_bytes = t * 288
        out.append(f"""func.def @{name}.init(%input:buffer) {{
  %o = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %end = index.constant {active*256} : index
  %x = buffer.view %input[%o] : buffer -> view<{t*256}xf32>
  %value = scalar.constant 127.0 : f32
  scf.for %i = [%zero to %end step %one] {{
    view.store %value, %x[%i] : f32, view<{t*256}xf32>
  }}
  func.return
}}
command.def @{name}.actual() launch(%input:buffer,%count:buffer,%output:buffer) {{
  %T = index.constant {t} : index
  %K = index.constant 256 : index
  template.apply<@llm.projection.prepare_canonical_prefix>(%T,%K,%count,%input,%output) : (index,index,buffer,buffer,buffer)
  command.return
}}
func.def @{name}.expected(%input:buffer,%count:buffer,%output:buffer) {{
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %groups = index.constant {t*2} : index
  %active_groups = index.constant {active*2} : index
  %four = index.constant 4 : index
  %two = index.constant 2 : index
  %payload_begin = index.constant 16 : index
  %end = index.constant 144 : index
  %stride = index.constant 144 : offset
  scf.for %group = [%zero to %groups step %one] {{
    %base = index.scale %group, %stride : index, offset -> offset
    %live = index.cmp ult, %group, %active_groups : index
    %header = buffer.view %output[%base] : buffer -> view<8xf16>
    %bytes = buffer.view %output[%base] : buffer -> view<144xi8>
    %z = scalar.constant 0.0 : f16
    %d = scalar.constant 1.0 : f16
    %sum = scalar.constant 4064.0 : f16
    %scale = scf.select %live, %d, %z : f16
    %scaled_sum = scf.select %live, %sum, %z : f16
    scf.for %block = [%zero to %four step %one] {{
      %di = index.mul %block, %two : index
      %si = index.add %di, %one : index
      view.store %scale, %header[%di] : f16, view<8xf16>
      view.store %scaled_sum, %header[%si] : f16, view<8xf16>
    }}
    %code = scalar.constant 127 : i8
    %zero_code = scalar.constant 0 : i8
    %payload = scf.select %live, %code, %zero_code : i8
    scf.for %byte = [%payload_begin to %end step %one] {{
      view.store %payload, %bytes[%byte] : i8, view<144xi8>
    }}
  }}
  func.return
}}
check.scenario @{name} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-777.0) : tensor<{t*256}xf32>
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %owner = check.generate.fill value(-99) : tensor<{packed_bytes+32}xi8>
    %output = check.tensor.view %owner offset(16) : tensor<{packed_bytes+32}xi8> -> tensor<{packed_bytes}xi8>
    check.generate<@{name}.init>(%input) : (tensor<{t*256}xf32>) -> ()
    check.compare<@{name}.actual,@{name}.expected>(%input,%count,%output) : (tensor<{t*256}xf32>,tensor<3xi32>,tensor<{packed_bytes}xi8>) -> () {{
      check.expect.bitwise actual(%owner) expected(%owner) : tensor<{packed_bytes+32}xi8>
      check.expect.bitwise actual(%input) expected(%input) : tensor<{t*256}xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
    }}
  }}
  check.return
}}
""")
    return "".join(out)


def gated_cases():
    out = [
        "template.decl @llm.projection.gated_matrix(%T:index,%K:index,%N:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%gate:view<[%bytes]xi8>,%up:view<[%bytes]xi8>,%output:buffer)\n",
        "kernel.decl @llm.projection.active_swiglu(%T:index,%N:index) launch(%T:index,%N:index,%count:buffer,%gate:buffer,%up:buffer,%output:buffer)\n",
    ]
    for t, active in ((1, 0), (1, 1), (4, 1), (4, 4), (16, 0), (16, 15), (16, 16)):
        name = f"shared_gated_t{t}_a{active}"
        qkv = f"shared_qkv_t{t}_a{active}"
        wb, kwb, vwb = 256 * 144, 128 * 210, 128 * 144
        args = "%input:buffer,%count:buffer,%gate:buffer,%up:buffer,%output:buffer"
        for route in ("candidate", "baseline"):
            out.append(f"""command.def @{name}.{route}() launch({args}) {{
  %T = index.constant {t} : index
  %K = index.constant 256 : index
  %N = index.constant 256 : index
  %tag = index.constant 12 : index
  %wb = index.constant {wb} : index
  %o = index.constant 0 : offset
  %g = buffer.view %gate[%o] : buffer -> view<[%wb]xi8>
  %u = buffer.view %up[%o] : buffer -> view<[%wb]xi8>
""")
            if route == "candidate":
                out.append(
                    f"  template.apply<@llm.projection.gated_matrix>(%T,%K,%N,%tag,%wb,%count,%input,%g,%u,%output) : (index,index,index,index,index,buffer,buffer,view<[%wb]xi8>,view<[%wb]xi8>,buffer)\n"
                )
            else:
                out.append(f"""  %bytes = index.constant {t*256*4} : offset
  %a = buffer.alloca<global> align(256) %bytes : buffer
  %b = buffer.alloca<global> align(256) %bytes : buffer
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%wb,%count,%input,%g,%a) : (index,index,index,index,index,buffer,buffer,view<[%wb]xi8>,buffer)
  template.apply<@llm.projection.matrix>(%T,%K,%N,%tag,%wb,%count,%input,%u,%b) : (index,index,index,index,index,buffer,buffer,view<[%wb]xi8>,buffer)
  kernel.launch @llm.projection.active_swiglu[%T,%N](%T,%N,%count,%a,%b,%output) : [index,index](index,index,buffer,buffer,buffer,buffer)
""")
            out.append("  command.return\n}\n")
        out.append(f"""func.def @{name}.expected({args}) {{
  %o = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %A = index.constant {active} : index
  %N = index.constant 256 : index
  %out = buffer.view %output[%o] : buffer -> view<{t}x256xf32>
  %unit = scalar.constant 48768.0 : f32
  %z = scalar.constant 0.0 : f32
  scf.for %row = [%zero to %A step %one] {{
    %phase = index.rem %row, %three : index
    %factor = index.add %phase, %one : index
    %factor32 = index.cast %factor : index to i32
    %f = scalar.sitofp %factor32 : i32 to f32
    %g = scalar.mulf %f, %unit : f32
    %square = scalar.mulf %g, %g : f32
    %parity = index.rem %row, %two : index
    %positive = index.cmp eq, %parity, %zero : index
    // Positive gates saturate SiLU to g; negative gates underflow to -0.
    %value = scf.select %positive, %square, %z : f32
    scf.for %column = [%zero to %N step %one] {{
      view.store %value, %out[%row,%column] : f32, view<{t}x256xf32>
    }}
  }}
  func.return
}}
""")
        for route in ("candidate", "baseline"):
            scenario = f"{name}_{route}"
            out.append(f"""check.scenario @{scenario} {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input = check.generate.fill value(-777.0) : tensor<{t*256}xf32>
    %count = check.generate.fill value(0) : tensor<3xi32>
    %gate = check.generate.fill value(0) : tensor<{wb}xi8>
    %up = check.generate.fill value(0) : tensor<{wb}xi8>
    %kw = check.generate.fill value(0) : tensor<{kwb}xi8>
    %vw = check.generate.fill value(0) : tensor<{vwb}xi8>
    %owner = check.generate.fill value(-991.0) : tensor<{t*256+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{t*256+8}xf32> -> tensor<{t*256}xf32>
    check.generate<@{qkv}.init>(%input,%count,%gate,%kw,%vw) : (tensor<{t*256}xf32>,tensor<3xi32>,tensor<{wb}xi8>,tensor<{kwb}xi8>,tensor<{vwb}xi8>) -> ()
    check.generate<@{qkv}.init>(%input,%count,%up,%kw,%vw) : (tensor<{t*256}xf32>,tensor<3xi32>,tensor<{wb}xi8>,tensor<{kwb}xi8>,tensor<{vwb}xi8>) -> ()
    check.compare<@{name}.{route},@{name}.expected>(%input,%count,%gate,%up,%output) : (tensor<{t*256}xf32>,tensor<3xi32>,tensor<{wb}xi8>,tensor<{wb}xi8>,tensor<{t*256}xf32>) -> () {{
      check.expect.bitwise actual(%owner) expected(%owner) : tensor<{t*256+8}xf32>
      check.expect.bitwise actual(%input) expected(%input) : tensor<{t*256}xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
      check.expect.bitwise actual(%gate) expected(%gate) : tensor<{wb}xi8>
      check.expect.bitwise actual(%up) expected(%up) : tensor<{wb}xi8>
    }}
  }}
  check.return
}}
check.benchmark<@{scenario}> @bench_{scenario}
""")
    return "".join(out)


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(emit())
