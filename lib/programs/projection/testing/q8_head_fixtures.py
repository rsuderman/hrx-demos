#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Shared output motif: Q8/raw-half, last row, independent tables and tails."""

from pathlib import Path
import sys

text = ["""// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
config.def @ggml.linear_q4k_q8_1_x4.token_capacity = 1 : index
config.def @ggml.linear_q4k_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q4_0_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_q4_0_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q4_1_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_q4_1_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_iq4_xs_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_iq4_xs_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_iq4_nl_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_iq4_nl_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_iq3_s_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_iq3_s_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q3k_q8_1_x4.token_capacity = 512 : index
config.def @ggml.linear_q3k_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q5_0_q8_1_x4.token_capacity = 1 : index
config.def @ggml.linear_q5_0_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q5k_q8_1_x4.token_capacity = 1 : index
config.def @ggml.linear_q5k_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q6k_q8_1_x4.token_capacity = 1 : index
config.def @ggml.linear_q6k_q8_1_x4.output_capacity = 1 : index
config.def @ggml.linear_q8_0_q8_1_x4.token_capacity = 1 : index
config.def @ggml.linear_q8_0_q8_1_x4.output_capacity = 128256 : index
config.def @ggml.quantize_q8_1_x4.group_capacity = 2 : index
config.def @qwen38.greedy_argmax.output_capacity = 262208 : index
kernel.decl @llm.decoder.output_norm_last(%T:index,%E:index) launch(%T:index,%E:index,%begin:index,%origin:offset,%epsilon:f32,%count:buffer,%input:buffer,%gamma:buffer,%output:buffer)
template.decl @llm.output.norm_last(%T:index,%begin:index,%origin:offset,%E:index,%count:buffer,%input:buffer,%gamma:view<[%E]xf32>,%output:buffer)
template.def<@llm.output.norm_last> @q8_head.norm_last(%T:index,%begin:index,%origin:offset,%E:index,%count:buffer,%input:buffer,%gamma:view<[%E]xf32>,%output:buffer) {
  %epsilon = scalar.constant 0.0 : f32
  kernel.launch @llm.decoder.output_norm_last[%T,%E](%T,%E,%begin,%origin,%epsilon,%count,%input,%gamma,%output) : [index,index](index,index,index,offset,f32,buffer,buffer,view<[%E]xf32>,buffer)
  template.return
}
template.decl @llm.output.last_greedy(%T:index,%begin:index,%origin:offset,%E:index,%V:index,%B:index,%policy:index,%tag:index,%input:buffer,%count:buffer,%gamma:view<[%E]xf32>,%weight:view<[%B]xi8>,%token:buffer)
func.def @q8_head.set_row(%token:index,%code:i8,%weight:buffer) {
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %three = index.constant 3 : index
  %blocks = index.constant 8 : index
  %payload = index.constant 32 : index
  %rowbytes = index.constant 272 : offset
  %blockbytes = index.constant 34 : offset
  %two = index.constant 2 : offset
  %row = index.scale %token,%rowbytes : index,offset -> offset
  %unit = scalar.constant 0.25 : f32
  scf.for %block = [%zero to %blocks step %one] {
    %add = index.scale %block,%blockbytes : index,offset -> offset
    %origin = index.add %row,%add : offset
    %codes_origin = index.add %origin,%two : offset
    %scale = buffer.view %weight[%origin] : buffer -> view<1xf16>
    %codes = buffer.view %weight[%codes_origin] : buffer -> view<32xi8>
    %phase = index.rem %block,%three : index
    %factor = index.add %phase,%one : index
    %factor32 = index.cast %factor : index to i32
    %factor_f = scalar.uitofp %factor32 : i32 to f32
    %d32 = scalar.mulf %factor_f,%unit : f32
    %d16 = scalar.fptrunc %d32 : f32 to f16
    view.store %d16,%scale[0] : f16,view<1xf16>
    scf.for %lane = [%zero to %payload step %one] { view.store %code,%codes[%lane] : i8,view<32xi8> }
  }
  func.return
}
func.def @raw_head.set_row(%token:index,%code:i8,%weight:buffer) {
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %width = index.constant 256 : index
  %rowbytes = index.constant 512 : offset
  %origin = index.scale %token,%rowbytes : index,offset -> offset
  %w = buffer.view %weight[%origin] : buffer -> view<256xf16>
  %value = scalar.sitofp %code : i8 to f32
  %half = scalar.fptrunc %value : f32 to f16
  scf.for %column = [%zero to %width step %one] { view.store %half,%w[%column] : f16,view<256xf16> }
  func.return
}
"""]
args = "%input:buffer,%count:buffer,%gamma:buffer,%table_a:buffer,%table_b:buffer,%normalized:buffer,%token_a:buffer,%token_b:buffer"
cases = [(8, v, a, 0, 0) for v, a in ((256, 1), (256, 2), (128256, 1), (128256, 16))]
cases += [
    (1, v, a, 0, 0)
    for v, a in ((257, 1), (257, 16), (128256, 1), (128256, 16), (262208, 1))
]
cases += [(1, 257, 2, 3, 16)]
for tag, vocab, active, begin, count_origin in cases:
    name = f"head_tag{tag}_v{vocab}_a{active}"
    if begin or count_origin:
        name += f"_b{begin}_o{count_origin}"
    count_words = count_origin // 4 + 3
    size, elements = vocab * (512 if tag == 1 else 272), 17 * 256
    setter = "raw_head" if tag == 1 else "q8_head"
    policy = 2 if tag == 1 else 1
    sign = 1 if (begin + active) % 2 else -1
    winner_a, winner_b = (vocab - 2, vocab - 4) if sign > 0 else (41, 53)
    text.append(
        f"func.def @{name}.init(%input:buffer,%table_a:buffer,%table_b:buffer) {{\n"
    )
    for table, ids, codes in (
        ("table_a", (vocab - 2, vocab - 1, 41, 43), (3, 3, -5, -5)),
        ("table_b", (vocab - 4, vocab - 3, 53, 59), (7, 7, -11, -11)),
    ):
        for i, (token, code) in enumerate(zip(ids, codes)):
            text.append(
                f"  %{table}_id{i} = index.constant {token} : index\n  %{table}_code{i} = scalar.constant {code} : i8\n  func.call @{setter}.set_row(%{table}_id{i},%{table}_code{i},%{table}) : (index,i8,buffer) -> ()\n"
            )
    text.append(
        f"  %origin = index.constant 0 : offset\n  %zero = index.constant 0 : index\n  %one = index.constant 1 : index\n  %E = index.constant 256 : index\n  %x = buffer.view %input[%origin] : buffer -> view<17x256xf32>\n  %nanbits = scalar.constant 2143289344 : i32\n  %nan = scalar.bitcast %nanbits : i32 to f32\n"
    )
    for row in range(17):
        text.append(
            f"  %v{row} = scalar.constant {float((1 if row%2==0 else -1)*(row+1))} : f32\n  scf.for %c = [%zero to %E step %one] {{ view.store {'%v'+str(row) if row<16 else '%nan'},%x[{row},%c] : f32,view<17x256xf32> }}\n"
        )
    text.append("  func.return\n}\n")
    count_setup = ""
    if count_origin:
        text.append(f"""func.def @{name}.count_init(%count:buffer) {{
  %origin = index.constant {count_origin} : offset
  %value = scalar.constant {active} : i32
  %words = buffer.view %count[%origin] : buffer -> view<3xi32>
  view.store %value,%words[0] : i32,view<3xi32>
  func.return
}}
""")
        count_setup = f"check.generate<@{name}.count_init>(%count) : (tensor<{count_words}xi32>) -> ()"
    text.append(f"""command.def @{name}.actual() launch({args}) {{
  %T = index.constant 17 : index
  %E = index.constant 256 : index
  %V = index.constant {vocab} : index
  %B = index.constant {size} : index
  %zero = index.constant 0 : index
  %begin = index.constant {begin} : index
  %origin = index.constant 0 : offset
  %count_origin = index.constant {count_origin} : offset
  %policy = index.constant {policy} : index
  %tag = index.constant {tag} : index
  %g = buffer.view %gamma[%origin] : buffer -> view<[%E]xf32>
  %a = buffer.view %table_a[%origin] : buffer -> view<[%B]xi8>
  %b = buffer.view %table_b[%origin] : buffer -> view<[%B]xi8>
  template.apply<@llm.output.norm_last>(%T,%begin,%count_origin,%E,%count,%input,%g,%normalized) : (index,index,offset,index,buffer,buffer,view<[%E]xf32>,buffer)
  template.apply<@llm.output.last_greedy>(%T,%begin,%count_origin,%E,%V,%B,%policy,%tag,%input,%count,%g,%a,%token_a) : (index,index,offset,index,index,index,index,index,buffer,buffer,view<[%E]xf32>,view<[%B]xi8>,buffer)
  template.apply<@llm.output.last_greedy>(%T,%begin,%count_origin,%E,%V,%B,%policy,%tag,%input,%count,%g,%b,%token_b) : (index,index,offset,index,index,index,index,index,buffer,buffer,view<[%E]xf32>,view<[%B]xi8>,buffer)
  command.return
}}
func.def @{name}.expected({args}) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %E = index.constant 256 : index
  %n = buffer.view %normalized[%origin] : buffer -> view<256xf32>
  %a = buffer.view %token_a[%origin] : buffer -> view<1xi32>
  %b = buffer.view %token_b[%origin] : buffer -> view<1xi32>
  %unit = scalar.constant {float(sign)} : f32
  %id_a = scalar.constant {winner_a} : i32
  %id_b = scalar.constant {winner_b} : i32
  view.store %id_a,%a[0] : i32,view<1xi32>
  view.store %id_b,%b[0] : i32,view<1xi32>
  scf.for %c = [%zero to %E step %one] {{ view.store %unit,%n[%c] : f32,view<256xf32> }}
  func.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input_owner = check.generate.fill value(-777.0) : tensor<{elements+8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{elements+8}xf32> -> tensor<{elements}xf32>
    %a_owner = check.generate.fill value(0) : tensor<{size+512}xi8>
    %table_a = check.tensor.view %a_owner offset(256) : tensor<{size+512}xi8> -> tensor<{size}xi8>
    %b_owner = check.generate.fill value(0) : tensor<{size+512}xi8>
    %table_b = check.tensor.view %b_owner offset(256) : tensor<{size+512}xi8> -> tensor<{size}xi8>
    check.generate<@{name}.init>(%input,%table_a,%table_b) : (tensor<{elements}xf32>,tensor<{size}xi8>,tensor<{size}xi8>) -> ()
    %count = check.generate.fill value({-999 if count_origin else active}) : tensor<{count_words}xi32>
    {count_setup}
    %gamma = check.generate.fill value(1.0) : tensor<256xf32>
    %normal_owner = check.generate.fill value(-99.0) : tensor<264xf32>
    %normalized = check.tensor.view %normal_owner offset(16) : tensor<264xf32> -> tensor<256xf32>
    %normal_prefix = check.tensor.view %normal_owner offset(0) : tensor<264xf32> -> tensor<4xf32>
    %normal_suffix = check.tensor.view %normal_owner offset(1040) : tensor<264xf32> -> tensor<4xf32>
    %token_owner = check.generate.fill value(-77) : tensor<10xi32>
    %token_a = check.tensor.view %token_owner offset(16) : tensor<10xi32> -> tensor<1xi32>
    %token_b = check.tensor.view %token_owner offset(20) : tensor<10xi32> -> tensor<1xi32>
    check.compare<@{name}.actual,@{name}.expected>(%input,%count,%gamma,%table_a,%table_b,%normalized,%token_a,%token_b) : (tensor<{elements}xf32>,tensor<{count_words}xi32>,tensor<256xf32>,tensor<{size}xi8>,tensor<{size}xi8>,tensor<256xf32>,tensor<1xi32>,tensor<1xi32>) -> () {{
      check.expect.bitwise actual(%token_owner) expected(%token_owner) : tensor<10xi32>
      check.expect.approximate actual(%normal_owner) expected(%normal_owner) atol(0.00001) rtol(0.00001) nan(same) : tensor<264xf32>,tensor<264xf32>
      check.expect.bitwise actual(%normal_prefix) expected(%normal_prefix) : tensor<4xf32>
      check.expect.bitwise actual(%normal_suffix) expected(%normal_suffix) : tensor<4xf32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{elements+8}xf32>
      check.expect.bitwise actual(%a_owner) expected(%a_owner) : tensor<{size+512}xi8>
      check.expect.bitwise actual(%b_owner) expected(%b_owner) : tensor<{size+512}xi8>
      check.expect.bitwise actual(%count) expected(%count) : tensor<{count_words}xi32>
      check.expect.bitwise actual(%gamma) expected(%gamma) : tensor<256xf32>
    }}
  }}
  check.return
}}
""")
Path(sys.argv[1]).write_text("".join(text))
