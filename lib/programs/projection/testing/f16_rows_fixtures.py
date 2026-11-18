#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Raw-half projection witnesses; no production decoder or captured GPU oracle.

Logical half bits cover negative/zero/subnormal weights and cancellation. F32
activations deliberately contain values lost by F16 or Q8 activation rounding.
Expected products and sums are rounded independently by the host IEEE F32 packer.
"""

from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


HALF_BITS = (0x3C00, 0xBC00, 0x3800, 0x0001, 0x8000, 0xC100)
ACTIVATIONS = (1 + 1 / 4096, 1.0, -3 / 8, 1024.0, -0.25, 0.125)


def expected(row, column):
    total = 0.0
    for slot, value in enumerate(ACTIVATIONS):
        bits = HALF_BITS[(slot + column) % len(HALF_BITS)]
        weight = struct.unpack("<e", struct.pack("<H", bits))[0]
        activation = f32((row + 1) * value)
        total = f32(total + f32(activation * weight))
    return total


def fixture(rows, width, columns, active, capacity=None):
    capacity = capacity or rows
    key = f"f16_r{rows}_k{width}_n{columns}_c{active}_p{capacity}".replace("-", "m")
    size = 2 * width * columns
    live = active if 0 < active <= rows else 0
    indexes = (0, 1, 2, 3, 4, width - 1)
    out = [f"func.def @{key}.init(%input:buffer,%weight:buffer) {{\n"]
    out.append(f"""  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero_h = scalar.constant 0.0 : f16
  %P = index.constant {capacity} : index
  %nan_bits = scalar.constant 2143289344 : i32
  %nan = scalar.bitcast %nan_bits : i32 to f32
  %x = buffer.view %input[%origin] : buffer -> view<{capacity}x{width}xf32>
  %w = buffer.view %weight[%origin] : buffer -> view<{columns}x{width}xf16>
  scf.for %r = [%zero to %P step %one] {{ scf.for %k = [%zero to %K step %one] {{ view.store %nan,%x[%r,%k] : f32,view<{capacity}x{width}xf32> }} }}
""")
    for row in range(live):
        out.append(
            f"  scf.for %k = [%zero to %K step %one] {{ view.store %zero_f,%x[{row},%k] : f32,view<{capacity}x{width}xf32> }}\n"
        )
        for slot, index in enumerate(indexes):
            out.append(
                f"  %x{row}_{slot} = scalar.constant {f32((row + 1) * ACTIVATIONS[slot])!r} : f32\n  view.store %x{row}_{slot},%x[{row},{index}] : f32,view<{capacity}x{width}xf32>\n"
            )
    out.append(
        f"  scf.for %c = [%zero to %N step %one] {{ scf.for %k = [%zero to %K step %one] {{ view.store %zero_h,%w[%c,%k] : f16,view<{columns}x{width}xf16> }} }}\n"
    )
    for slot, bits in enumerate(HALF_BITS):
        signed = bits if bits < 32768 else bits - 65536
        out.append(
            f"  %bits{slot} = scalar.constant {signed} : i16\n  %h{slot} = scalar.bitcast %bits{slot} : i16 to f16\n"
        )
    for column in range(columns):
        for slot, index in enumerate(indexes):
            out.append(
                f"  view.store %h{(slot + column) % 6},%w[{column},{index}] : f16,view<{columns}x{width}xf16>\n"
            )
    out.append("  func.return\n}\n")
    args = "%input:buffer,%weight:buffer,%count:buffer,%packed:buffer,%output:buffer"
    out.append(f"""command.def @{key}.actual() launch({args}) {{
  %R = index.constant {rows} : index
  %C = index.constant {capacity} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %B = index.constant {size} : index
  %tag = index.constant 1 : index
  %origin = index.constant 0 : offset
  %w = buffer.view %weight[%origin] : buffer -> view<[%B]xi8>
  template.apply<@llm.projection.canonical_rows_prepacked>(%R,%C,%K,%N,%tag,%B,%count,%input,%packed,%w,%output) : (index,index,index,index,index,index,buffer,buffer,buffer,view<[%B]xi8>,buffer)
  command.return
}}
func.def @{key}.expected({args}) {{
  %origin = index.constant 0 : offset
  %y = buffer.view %output[%origin] : buffer -> view<{rows}x{columns}xf32>
""")
    for row in range(rows):
        for column in range(columns):
            value = expected(row, column) if row < live else 0.0
            out.append(
                f"  %v{row}_{column} = scalar.constant {value!r} : f32\n  view.store %v{row}_{column},%y[{row},{column}] : f32,view<{rows}x{columns}xf32>\n"
            )
    out.append(f"""  func.return
}}
check.scenario @{key}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %input_owner = check.generate.fill value(-777.0) : tensor<{capacity * width + 8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{capacity * width + 8}xf32> -> tensor<{capacity * width}xf32>
    %weight_owner = check.generate.fill value(-555.0) : tensor<{width * columns + 16}xf16>
    %weight = check.tensor.view %weight_owner offset(16) : tensor<{width * columns + 16}xf16> -> tensor<{width * columns}xf16>
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %packed = check.generate.fill value(-99) : tensor<16xi8>
    %output_owner = check.generate.fill value(-888.0) : tensor<{capacity * columns + 8}xf32>
    %output = check.tensor.view %output_owner offset(16) : tensor<{capacity * columns + 8}xf32> -> tensor<{capacity * columns}xf32>
    check.generate<@{key}.init>(%input,%weight) : (tensor<{capacity * width}xf32>,tensor<{width * columns}xf16>) -> ()
    check.compare<@{key}.actual,@{key}.expected>(%input,%weight,%count,%packed,%output) : (tensor<{capacity * width}xf32>,tensor<{width * columns}xf16>,tensor<3xi32>,tensor<16xi8>,tensor<{capacity * columns}xf32>) -> () {{
      check.expect.bitwise actual(%output_owner) expected(%output_owner) : tensor<{capacity * columns + 8}xf32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{capacity * width + 8}xf32>
      check.expect.bitwise actual(%weight_owner) expected(%weight_owner) : tensor<{width * columns + 16}xf16>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
      check.expect.bitwise actual(%packed) expected(%packed) : tensor<16xi8>
    }}
  }}
  check.return
}}
""")
    return "".join(out)


declaration = "template.decl @llm.projection.canonical_rows_prepacked(%rows:index,%capacity:index,%width:index,%columns:index,%tag:index,%bytes:index,%count:buffer,%input:buffer,%packed:buffer,%weight:view<[%bytes]xi8>,%output:buffer)\n"
cases = [
    (r, 1024, 16, n)
    for r in (1, 8, 16, 17)
    for n in sorted({-1, 0, 1, r // 2, r, r + 1})
]
text = declaration + "".join(fixture(*case) for case in cases)
text += fixture(1, 7, 257, 1)
text += fixture(8, 7, 3, 1, capacity=17)
Path(sys.argv[1]).write_text(text)
