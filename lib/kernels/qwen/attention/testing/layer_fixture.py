# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
"""Bounded full-layer fixture from fresh native command parameter reflection.

The uniform nonzero GGML blocks represent constant matrix rows. The scalar
oracle computes quantized dot products, learned norms, sigmoid, NeoX rotary,
F16 cache publication, output projection, and exactly one residual/FFN pair.
General QK score/GQA and fragmented-map coverage lives in primitive tests.
"""
import argparse
import array
import math
from pathlib import Path
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def f16(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def npy(path, data, dtype, shape):
    header = repr(dict(descr=dtype, fortran_order=False, shape=shape)).encode()
    header += b' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + b'\n'
    path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + data)


def quant_block(kind, scale):
    d = struct.pack('<e', scale)
    if kind == 6:
        # q=33-32=1, signed group scales=1.
        return bytes([0x11]) * 128 + bytes([0xaa]) * 64 + bytes([1]) * 16 + d
    # Eight scale=1/min=0 groups, packed as canonical GGML K scales.
    scales = bytes([1] * 4 + [0] * 4 + [1] * 4)
    return d + b'\0\0' + scales + (bytes(32) if kind == 5 else b'') + bytes([0x11]) * 128


def dot(row, scale):
    """Independent Q8_1 block rounding and constant GGML K row dot product."""
    total = 0.0
    for start in range(0, len(row), 32):
        values = row[start:start + 32]
        d = f32(max(abs(x) for x in values) / 127)
        codes = [math.floor(f32(x / d) + .5) if d else 0 for x in values]
        total += sum(codes) * f16(d) * f16(scale)
    return f32(total)


def norm(row, gamma):
    inv = 1 / math.sqrt(sum(x*x for x in row) / len(row) + 1e-6)
    return [f32(x * inv * gamma) for x in row]


def build(reflection, destination, query_kind):
    destination.mkdir(parents=True, exist_ok=True)
    lines = [line.split('\t') for line in reflection.read_text().splitlines()]
    assert lines[0][0] == 'ROOT' and int(lines[0][3]) == 0
    size = int(lines[0][1])
    assert len(lines) == 12
    bank = bytearray(size)
    for key, offset, length, root in lines[1:]:
        offset, length = int(offset), int(length)
        assert root == '0' and key.startswith('blk.3.')
        role = key.removeprefix('blk.3.')
        if 'norm.weight' in role:
            payload = struct.pack('<f', 1.) * (length // 4)
        else:
            kind = (query_kind if role in ('attn_q.weight', 'attn_k.weight') else
                    6 if role in ('attn_v.weight', 'ffn_down.weight') else
                    4 if role in ('ffn_gate.weight', 'ffn_up.weight') else 5)
            scale = 1 / (2048 if role == 'attn_output.weight' else
                         3584 if role == 'ffn_down.weight' else 1024)
            block = quant_block(kind, scale)
            assert length % len(block) == 0, (key, length, len(block))
            payload = block * (length // len(block))
        bank[offset:offset+length] = payload
    npy(destination/'bank.npy', bank, '|i1', (size,))
    cases = []
    for count, gamma in [(1, 1.), (2, 1.), (2, .5)]:
        name = f'n{count}_gamma{int(2*gamma)}'
        case = destination/name
        case.mkdir(exist_ok=True)
        changed_bank = bytearray(bank)
        key, offset, length, _ = next(x for x in lines if x[0] == 'blk.3.attn_norm.weight')
        changed_bank[int(offset):int(offset)+int(length)] = struct.pack('<f', gamma)*1024
        npy(case/'bank.npy', changed_bank, '|i1', (size,))
        row = [1., 2.] * 512
        residual = row*4 + [-31.]*16
        expected = list(residual)
        projected = dot(norm(row, gamma), 1/1024)
        value = f16(projected)
        attended = f32(value / (1 + math.exp(-projected)))
        projected_attention = dot([attended]*2048, 1/2048)
        for r in range(4):
            first = [f32(x + (projected_attention if r < count else 0)) for x in row]
            gate_up = dot(norm(first, 1.), 1/1024)
            swiglu = f32(gate_up * gate_up / (1 + math.exp(-gate_up)))
            down = dot([swiglu]*3584, 1/3584)
            expected[r*1024:(r+1)*1024] = [f32(x+down) for x in first]
        # C=96, P=128, nonzero arena origin, reversed pages, ordinal zero.
        arena = [-7.] * (32 + 12*128*512 + 32)
        expected_arena = list(arena)
        key_norm = projected / math.sqrt(projected*projected + 1e-6)
        for r in range(count):
            physical = 64 + r
            for h in range(2):
                for lane in range(256):
                    k = key_norm
                    if lane < 64:
                        angle = r * 1e7 ** (-(lane % 32)/32)
                        k *= math.cos(angle) + (-1 if lane < 32 else 1)*math.sin(angle)
                    expected_arena[32 + physical*512 + h*256+lane] = f16(k)
                    expected_arena[32 + 128*512 + physical*512 + h*256+lane] = value
        epoch = [0]*1048
        epoch[:8] = [count, 1, 0, count, 0, 0, 0, -1]
        rows = bytearray(2056)
        struct.pack_into('<q', rows, 16, 64)
        struct.pack_into('<ii', rows, 2048, 1, 0)
        npy(case/'residual.npy', array.array('f', residual).tobytes(), '<f4', (4112,))
        npy(case/'expected.npy', array.array('f', expected).tobytes(), '<f4', (4112,))
        npy(case/'epoch.npy', array.array('i', epoch).tobytes(), '<i4', (1048,))
        npy(case/'rows.npy', rows, '|i1', (2056,))
        npy(case/'cache.npy', b''.join(struct.pack('<e', x) for x in arena), '<f2', (len(arena),))
        npy(case/'expected-cache.npy', b''.join(struct.pack('<e', x) for x in expected_arena), '<f2', (len(arena),))
        cases.append((name, case, len(arena), count))
    emit_check(destination/'check.loom', size, cases)
    build_chain(destination, size, lines, bank)


def emit_check(path, size, cases):
    cache = cases[0][2]
    text = f'''// Generated bounded fixture. Offsets come only from fresh native reflection.
command.decl @attention_layer() launch(%bank: buffer, %unused: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %cache: buffer)
command.def @layer_actual() launch(%bank: buffer, %unused: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %cache: buffer, %expected: buffer, %expected_cache: buffer) {{
  command.launch @attention_layer(%bank, %unused, %residual, %epoch, %rows, %cache) : (buffer, buffer, buffer, buffer, buffer, buffer)
  command.return
}}
func.def @layer_expected(%bank: buffer, %unused: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %cache: buffer, %expected: buffer, %expected_cache: buffer) {{
  %origin = index.constant 0 : offset
  %a = buffer.view %residual[%origin] : buffer -> view<4112xf32>
  %b = buffer.view %expected[%origin] : buffer -> view<4112xf32>
  %c = buffer.view %cache[%origin] : buffer -> view<{cache}xf16>
  %d = buffer.view %expected_cache[%origin] : buffer -> view<{cache}xf16>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %residual_end = index.constant 4112 : index
  %cache_end = index.constant {cache} : index
  scf.for %i = [%zero to %residual_end step %one] {{
    %value = view.load %b[%i] : view<4112xf32> -> f32
    view.store %value, %a[%i] : f32, view<4112xf32>
  }}
  scf.for %i = [%zero to %cache_end step %one] {{
    %value = view.load %d[%i] : view<{cache}xf16> -> f16
    view.store %value, %c[%i] : f16, view<{cache}xf16>
  }}
  func.return
}}
'''
    types = f'tensor<{size}xi8>, tensor<1xi8>, tensor<4112xf32>, tensor<1048xi32>, tensor<2056xi8>, tensor<{cache}xf16>, tensor<4112xf32>, tensor<{cache}xf16>'
    for name, case, _, count in cases:
        text += f'check.scenario @whole_layer_{name} {{\n  check.trial[1](%trial: index, %entropy: check.entropy) {{\n'
        for var, file, tensor in [('bank','bank',f'{size}xi8'), ('residual','residual','4112xf32'), ('epoch','epoch','1048xi32'), ('rows','rows','2056xi8'), ('cache','cache',f'{cache}xf16'), ('expected','expected','4112xf32'), ('expected_cache','expected-cache',f'{cache}xf16')]:
            text += f'    %{var} = check.file.read.npy path("{case/file}.npy") : tensor<{tensor}>\n'
        protected = [(32, 64*512), (32+(64+count)*512, (64-count)*512),
                     (32+128*512, 64*512),
                     (32+(128+64+count)*512, (64-count)*512)]
        for i, (start, length) in enumerate(protected):
            text += f'    %inactive{i} = check.tensor.view %cache offset({start*2}) : tensor<{cache}xf16> -> tensor<{length}xf16>\n'
        text += f'''    %unused = check.generate.fill value(77) : tensor<1xi8>
    %outside = check.tensor.view %residual offset(16384) : tensor<4112xf32> -> tensor<16xf32>
    %other_planes = check.tensor.view %cache offset(262208) : tensor<{cache}xf16> -> tensor<{cache-131104}xf16>
    %prefix = check.tensor.view %cache offset(0) : tensor<{cache}xf16> -> tensor<32xf16>
    check.compare<@layer_actual, @layer_expected>(%bank, %unused, %residual, %epoch, %rows, %cache, %expected, %expected_cache) : ({types}) -> () {{
      check.expect.approximate actual(%residual) expected(%residual) atol(0.004) rtol(0.004) nan(same) : tensor<4112xf32>, tensor<4112xf32>
      check.expect.approximate actual(%cache) expected(%cache) atol(0.002) rtol(0.002) nan(same) : tensor<{cache}xf16>, tensor<{cache}xf16>
      check.expect.bitwise actual(%outside) expected(%outside) : tensor<16xf32>
      check.expect.bitwise actual(%other_planes) expected(%other_planes) : tensor<{cache-131104}xf16>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<32xf16>
      check.expect.bitwise actual(%bank) expected(%bank) : tensor<{size}xi8>
      check.expect.bitwise actual(%unused) expected(%unused) : tensor<1xi8>
      check.expect.bitwise actual(%epoch) expected(%epoch) : tensor<1048xi32>
      check.expect.bitwise actual(%rows) expected(%rows) : tensor<2056xi8>
'''
        for i, (_, length) in enumerate(protected):
            text += f'      check.expect.bitwise actual(%inactive{i}) expected(%inactive{i}) : tensor<{length}xf16>\n'
        text += "    }\n  }\n  check.return\n}\n"
    path.write_text(text)


def rotate(value, position):
    result = [value] * 256
    for lane in range(64):
        angle = position * 1e7 ** (-(lane % 32)/32)
        result[lane] = f32(value * (math.cos(angle) + (-1 if lane < 32 else 1)*math.sin(angle)))
    return result


def simulate_span(residual, cache, count, position, begin):
    projections = [dot(norm(residual[r*1024:(r+1)*1024], 1.), 1/1024) for r in range(4)]
    def address(p, head, lane, value=False):
        physical = (1 if p < 64 else 0)*64 + p%64
        return 32 + (128*512 if value else 0) + physical*512 + head*256+lane
    for i in range(count):
        p, row = position+i, begin+i
        projected = projections[row]
        normalized = .25 * projected / math.sqrt(projected*projected + 1e-6)
        key = rotate(normalized, p)
        for h in range(2):
            for lane in range(256):
                cache[address(p,h,lane)] = f16(key[lane])
                cache[address(p,h,lane,True)] = f16(projected)
    for row in range(4):
        attended = [0.] * 2048
        if begin <= row < begin+count:
            p = position + row-begin
            projected = projections[row]
            query = rotate(.25 * projected / math.sqrt(projected*projected + 1e-6), p)
            gate = 1/(1+math.exp(-projected))
            for head in range(8):
                kv = head//4
                scores = [sum(query[lane]*cache[address(k,kv,lane)] for lane in range(256))*.0625 for k in range(p+1)]
                maximum = max(scores)
                weights = [math.exp(score-maximum) for score in scores]
                denominator = sum(weights)
                for lane in range(256):
                    attended[head*256+lane] = f32(gate*sum(w*cache[address(k,kv,lane,True)] for k,w in enumerate(weights))/denominator)
        projected_attention = dot(attended, 1/2048)
        first = [f32(x+projected_attention) for x in residual[row*1024:(row+1)*1024]]
        gate_up = dot(norm(first, 1.), 1/1024)
        swiglu = f32(gate_up*gate_up/(1+math.exp(-gate_up)))
        down = dot([swiglu]*3584, 1/3584)
        residual[row*1024:(row+1)*1024] = [f32(x+down) for x in first]


def build_chain(destination, size, reflection, bank):
    # All old keys/values are initialized history, not invented zero-cache reads.
    cache = [-7.] * (32 + 12*128*512 + 32)
    for p in range(63):
        for head in range(2):
            for lane in range(256):
                physical = 64+p
                cache[32+physical*512+head*256+lane] = 0.
                cache[32+128*512+physical*512+head*256+lane] = f16((1+p%3)/8+head/16)
    residual = [1.,2.]*2048+[-31.]*16
    expected, expected_cache = list(residual), list(cache)
    steps = [(2,63,1),(2,65,0),(1,67,2),(1,0,0)]
    transcript = []
    for count, position, begin in steps:
        simulate_span(expected, expected_cache, count, position, begin)
        transcript.extend(expected[:4096])
    transcript.extend([-53.]*16)
    case=destination/'continuation'
    case.mkdir(exist_ok=True)
    # Small learned Q/K gamma keeps historical and newly appended softmax
    # terms significant instead of saturating attention onto the current row.
    chain_bank = bytearray(bank)
    for key,offset,length,_ in reflection[1:]:
        if key in ('blk.3.attn_q_norm.weight', 'blk.3.attn_k_norm.weight'):
            offset,length=int(offset),int(length)
            chain_bank[offset:offset+length]=struct.pack('<f',.25)*(length//4)
    npy(case/'bank.npy',chain_bank,'|i1',(size,))
    for name,data in [('cache',cache),('expected-cache',expected_cache)]:
        npy(case/(name+'.npy'), b''.join(struct.pack('<e',x) for x in data), '<f2',(len(data),))
    for name,data in [('residual',residual),('expected',expected),('expected-transcript',transcript)]:
        npy(case/(name+'.npy'),array.array('f',data).tobytes(),'<f4',(len(data),))
    epoch=[0]*1048
    epoch[:8]=[3,1,0,2,63,1,0,-1]
    npy(case/'epoch.npy',array.array('i',epoch).tobytes(),'<i4',(1048,))
    emit_chain_check(destination/'chain.loom',destination,case,size,len(cache),steps)


def emit_chain_check(path, destination, case, size, cache, steps):
    # Four complete calls reuse the actual session-emitted native entrypoint.
    text='command.decl @attention_layer() launch(%bank: buffer, %unused: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %cache: buffer)\n'
    for step,(count,position,begin) in enumerate(steps):
        text+=f'func.def @chain_metadata{step}(%epoch: buffer) {{\n  %origin = index.constant 0 : offset\n  %e = buffer.view %epoch[%origin] : buffer -> view<1048xi32>\n'
        for i,value in enumerate([begin+count,1,0,count,position,begin,0,-1]):
            text+=f'  %i{i} = index.constant {i} : index\n  %v{i} = scalar.constant {value} : i32\n  view.store %v{i}, %e[%i{i}] : i32, view<1048xi32>\n'
        text+='  func.return\n}\n'
        text+=f'''kernel.def @chain_step{step}() {{
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
}} launch(%epoch: buffer) {{
  func.call @chain_metadata{step}(%epoch) : (buffer) -> ()
  kernel.return
}}
kernel.def @chain_snapshot{step}() {{
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
}} launch(%residual: buffer, %transcript: buffer) {{
  %origin = index.constant 0 : offset
  %snapshot_origin = index.constant {step*16384} : offset
  %a = buffer.view %residual[%origin] : buffer -> view<4096xf32>
  %b = buffer.view %transcript[%snapshot_origin] : buffer -> view<4096xf32>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %end = index.constant 4096 : index
  scf.for %i = [%zero to %end step %one] {{
    %value = view.load %a[%i] : view<4096xf32> -> f32
    view.store %value, %b[%i] : f32, view<4096xf32>
  }}
  kernel.return
}}
'''
    signature='%bank: buffer, %unused: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %cache: buffer, %expected: buffer, %expected_cache: buffer, %transcript: buffer, %expected_transcript: buffer'
    text+=f'command.def @chain_actual() launch({signature}) {{\n'
    for step in range(4):
        text+=f'''  kernel.launch @chain_step{step}[](%epoch) : [](buffer)
  command.launch @attention_layer(%bank, %unused, %residual, %epoch, %rows, %cache) : (buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @chain_snapshot{step}[](%residual, %transcript) : [](buffer, buffer)
'''
    text+='  command.return\n}\n'
    text+=f'func.def @chain_expected({signature}) {{\n  %origin = index.constant 0 : offset\n  %zero = index.constant 0 : index\n  %one = index.constant 1 : index\n'
    for suffix,a,b,length,dtype in [('r','residual','expected',4112,'f32'),('c','cache','expected_cache',cache,'f16'),('t','transcript','expected_transcript',16400,'f32')]:
        text+=f'''  %a{suffix} = buffer.view %{a}[%origin] : buffer -> view<{length}x{dtype}>
  %b{suffix} = buffer.view %{b}[%origin] : buffer -> view<{length}x{dtype}>
  %end{suffix} = index.constant {length} : index
  scf.for %i = [%zero to %end{suffix} step %one] {{
    %v = view.load %b{suffix}[%i] : view<{length}x{dtype}> -> {dtype}
    view.store %v, %a{suffix}[%i] : {dtype}, view<{length}x{dtype}>
  }}
'''
    text+='  func.call @chain_metadata3(%epoch) : (buffer) -> ()\n  func.return\n}\ncheck.scenario @whole_layer_dependent_page_crossing_continuation_reset {\n  check.trial[1](%trial: index, %entropy: check.entropy) {\n'
    entries=[('bank',case/'bank.npy',f'{size}xi8'),('residual',case/'residual.npy','4112xf32'),('epoch',case/'epoch.npy','1048xi32'),('rows',destination/'n1_gamma2/rows.npy','2056xi8'),('cache',case/'cache.npy',f'{cache}xf16'),('expected',case/'expected.npy','4112xf32'),('expected_cache',case/'expected-cache.npy',f'{cache}xf16'),('expected_transcript',case/'expected-transcript.npy','16400xf32')]
    for name,file,tensor in entries:
        text+=f'    %{name} = check.file.read.npy path("{file}") : tensor<{tensor}>\n'
    text+='    %unused = check.generate.fill value(77) : tensor<1xi8>\n    %transcript = check.generate.fill value(-53.0) : tensor<16400xf32>\n'
    # Union of written logical positions is 0 and 63..67; protect the rest
    # exactly, including all initialized history and unmapped physical slack.
    protected=[('prefix',0,32),('old_history',32+65*512,62*512),('old_vhistory',32+(128+65)*512,62*512),('key_tail',32+4*512,60*512),('value_tail',32+(128+4)*512,60*512),('other_planes',32+256*512,cache-(32+256*512))]
    for name,start,length in protected:
        text+=f'    %{name} = check.tensor.view %cache offset({start*2}) : tensor<{cache}xf16> -> tensor<{length}xf16>\n'
    text+='    %outside = check.tensor.view %residual offset(16384) : tensor<4112xf32> -> tensor<16xf32>\n    %transcript_guard = check.tensor.view %transcript offset(65536) : tensor<16400xf32> -> tensor<16xf32>\n'
    types=f'tensor<{size}xi8>, tensor<1xi8>, tensor<4112xf32>, tensor<1048xi32>, tensor<2056xi8>, tensor<{cache}xf16>, tensor<4112xf32>, tensor<{cache}xf16>, tensor<16400xf32>, tensor<16400xf32>'
    text+=f'    check.compare<@chain_actual, @chain_expected>(%bank, %unused, %residual, %epoch, %rows, %cache, %expected, %expected_cache, %transcript, %expected_transcript) : ({types}) -> () {{\n'
    for name,tensor in [('residual','4112xf32'),('transcript','16400xf32'),('cache',f'{cache}xf16')]:
        text+=f'      check.expect.approximate actual(%{name}) expected(%{name}) atol(0.006) rtol(0.006) nan(same) : tensor<{tensor}>, tensor<{tensor}>\n'
    for name,tensor in [('outside','16xf32'),('transcript_guard','16xf32'),('bank',f'{size}xi8'),('unused','1xi8'),('epoch','1048xi32'),('rows','2056xi8')]+[(name,f'{length}xf16') for name,_,length in protected]:
        text+=f'      check.expect.bitwise actual(%{name}) expected(%{name}) : tensor<{tensor}>\n'
    text+='    }\n  }\n  check.return\n}\n'
    path.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reflection', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--query-kind', type=int, choices=(5, 6), required=True)
    args = parser.parse_args()
    build(args.reflection, args.destination.resolve(), args.query_kind)
