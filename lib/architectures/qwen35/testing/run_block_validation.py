# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Manual two-block actual-GGUF gate against external CPU callback captures.

Uses maintained numerical roots and reflected immutable GGUF ranges. No model
math, dequantizer, tokenizer, or production reference dependency is introduced.
Run only in the gate's isolated venv; the coordinator schedules the GPU lane.
"""
import argparse
import array
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import site
import struct
import subprocess
import sys

# Declared before executing the actual-weight gate. CPU K-block dot products
# may use Q8_K activation packing; Loom uses canonical per-32 Q8_1_x4 and F16
# WMMA attention. These are finite composition bounds, not bitwise math parity.
RELATIVE_L2_LIMIT = 0.02
ABSOLUTE_FLOOR = 0.05
MAX_RELATIVE_SCALE = 0.02


def digest(data):
    return hashlib.sha256(data).hexdigest()


def npy_header(length):
    text = repr({'descr': '|i1', 'fortran_order': False, 'shape': (length,)})
    text += ' ' * ((64 - (10 + len(text) + 1) % 64) % 64) + '\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text.encode()


def save(path, data):
    path.write_bytes(npy_header(len(data)) + data)


def load(path):
    data = path.read_bytes()
    assert data[:6] == b'\x93NUMPY'
    width = 2 if data[6] == 1 else 4
    length = int.from_bytes(data[8:8+width], 'little')
    info = ast.literal_eval(data[8+width:8+width+length].decode())
    assert info['descr'] in ('|i1', '<i1', '|u1', '<u1') and not info['fortran_order']
    return data[8+width+length:]


def floats(data):
    result = array.array('f')
    result.frombytes(data)
    assert all(math.isfinite(x) for x in result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('emitted', 'reference', 'gguf', 'reflector', 'runner', 'output', 'gpu-lock'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert sys.byteorder == 'little' and not os.environ.get('PYTHONPATH')
    assert not os.environ.get('PYTHONHOME')
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((args.emitted/'layout.json').read_text())
    reference = json.loads((args.reference/'capture.json').read_text())
    assert Path(reference['model']).resolve() == args.gguf.resolve()
    assert reference['n_gpu_layers'] == 0 and reference['kv_dtype'] == 'f16'
    assert geometry['hidden'] in (1024, 5120) and geometry['pool'] == 64
    assert [(x['step'], x['position'], x['token']) for x in reference['captures']] == [(0, 0, 100), (1, 1, 200), ('reset', 0, 100)]
    env = dict(os.environ, PYTHONNOUSERSITE='1', PARAMETER_GGUF=str(args.gguf))
    record = {'passed': False, 'geometry': geometry, 'reference': reference,
              'criteria': {'relative_l2': RELATIVE_L2_LIMIT,
                           'residual_delta_relative_l2': RELATIVE_L2_LIMIT,
                           'max_absolute_floor': ABSOLUTE_FLOOR,
                           'max_relative_scale': MAX_RELATIVE_SCALE,
                           'zero_norm_rule': 'zero reference norm requires exactly zero error'},
              'runs': [], 'comparisons': []}

    def publish():
        (args.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')

    def reference_tensor(capture, name):
        tensor = capture['tensors'][name]
        assert tensor['dtype'] == 'f32' and tensor['shape'] == [geometry['hidden']]
        path = (args.reference/tensor['path']).resolve()
        assert path.parent == args.reference.resolve()
        data = path.read_bytes()
        assert len(data) == geometry['hidden']*4
        floats(data)
        return data

    banks = []
    variants = {}
    for form in ('loom', 'loombc'):
        module = args.emitted/('configured_blocks.'+form)
        for block, layer, input_name, output_name in (
                ('gdn0', 0, 'model.input_embed', 'l_out-0'),
                ('attention3', 3, 'l_out-2', 'l_out-3')):
            root = 'validated.qwen.'+block
            work = args.output/(block+'-'+form)
            work.mkdir(exist_ok=True)
            reflected = work/'parameters.tsv'
            with reflected.open('w') as out, (work/'reflection.log').open('w') as err:
                subprocess.run([str(args.reflector), root, str(module)], env=env, stdout=out, stderr=err, check=True)
            roots = re.findall(r'ROOT (\d+) length=(\d+) alignment=(\d+)', (work/'reflection.log').read_text())
            assert len(roots) == 1 and roots[0][0] == '0'
            length = int(roots[0][1])
            fd = os.memfd_create('qwen-actual-block-bank', os.MFD_CLOEXEC)
            banks.append(fd)
            header = npy_header(length)
            os.ftruncate(fd, len(header)+length)
            os.pwrite(fd, header, 0)
            payloads = []
            with args.gguf.open('rb', buffering=0) as source:
                for line in reflected.read_text().splitlines():
                    name, offset, size, buffer, origin = line.split('\t')
                    assert int(buffer) == 0
                    source.seek(int(origin))
                    remaining, destination = int(size), len(header)+int(offset)
                    sha = hashlib.sha256()
                    while remaining:
                        data = source.read(min(remaining, 1024*1024))
                        assert data
                        sha.update(data)
                        assert os.pwrite(fd, data, destination) == len(data)
                        remaining -= len(data)
                        destination += len(data)
                    payloads.append({'name': name, 'bytes': int(size), 'source': int(origin), 'sha256': sha.hexdigest()})
            assert sorted(x['name'] for x in payloads) == sorted(geometry['bindings'][str(layer)])
            record['runs'].append({'block': block, 'form': form, 'module_sha256': digest(module.read_bytes()), 'payloads': payloads})
            payload_size = geometry['gdn_state_bytes'] if block == 'gdn0' else 2*geometry['pool']*geometry['plane_token_bytes']
            state_origin = geometry['state_origin']
            initial_state = bytes([91])*state_origin + bytes(payload_size) + bytes([91])*256
            rows = bytearray(geometry['map_origin']+4)
            struct.pack_into('<4q', rows, 0, state_origin, state_origin, state_origin, 0)
            struct.pack_into('<i', rows, geometry['map_origin'], 0)
            state = initial_state
            first = None
            for capture in reference['captures']:
                step = capture['step']
                case = work/str(step)
                case.mkdir(exist_ok=True)
                if step == 'reset':
                    state = initial_state
                reference_input = reference_tensor(capture, input_name)
                expected = reference_tensor(capture, output_name)
                residual = reference_input + struct.pack('<f', -31.)*((geometry['residual_bytes']-len(reference_input))//4+64)
                control = bytearray(4192)
                struct.pack_into('<8i', control, 0, 1, 1, 0, 1, capture['position'], 0, 0, -1)
                buffers = [residual, state, bytes(control), bytes([83])*geometry['capture_bytes'], bytes(rows)]
                for i, data in enumerate(buffers, 1):
                    save(case/f'arg{i}.npy', data)
                declarations = ','.join(f'%a{i}:buffer' for i in range(6))
                tensors = ','.join([f'tensor<{length}xi8>']+[f'tensor<{len(x)}xi8>' for x in buffers])
                fixture = f'command.decl @{root}() launch({declarations})\nfunc.def @noop({declarations}) {{ func.return }}\ncheck.scenario @actual_block {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n'
                fixture += f'  %a0 = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>\n'
                for i, data in enumerate(buffers, 1):
                    fixture += f'  %a{i} = check.file.read.npy path("{case/f"arg{i}.npy"}") : tensor<{len(data)}xi8>\n'
                fixture += f'  check.compare<@{root},@noop>(%a0,%a1,%a2,%a3,%a4,%a5) : ({tensors}) -> () {{\n'
                for i, size in ((0, length), (3, len(control)), (5, len(rows))):
                    fixture += f'   check.expect.bitwise actual(%a{i}) expected(%a{i}) : tensor<{size}xi8>\n'
                fixture += '  }\n }\n check.return\n}\n'
                (case/'check.loom').write_text(fixture)
                capture_dir = case/'capture'
                capture_dir.mkdir(exist_ok=True)
                command = [str(args.runner), str(case/'check.loom'), '--library='+str(module), '--case=@actual_block', '--device=amdgpu']
                with args.gpu_lock.open('a') as lock, (case/'execution.log').open('w') as log:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = subprocess.run(command, env=dict(env, LOOM_FFN_CAPTURE_DIR=str(capture_dir)), stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
                assert result.returncode == 0, case/'execution.log'
                output = load(capture_dir/'product0_call0_after_arg1.npy')
                next_state = load(capture_dir/'product0_call0_after_arg2.npy')
                assert output[len(reference_input):] == residual[len(reference_input):], (block, step, 'inactive residual')
                assert next_state[:state_origin] == initial_state[:state_origin]
                assert next_state[state_origin+payload_size:] == initial_state[state_origin+payload_size:]
                if block == 'attention3':
                    cursor = state_origin
                    permitted = [(state_origin+plane*geometry['pool']*geometry['plane_token_bytes']+capture['position']*geometry['plane_token_bytes'], geometry['plane_token_bytes']) for plane in (0, 1)]
                    for begin, size in permitted:
                        assert next_state[cursor:begin] == state[cursor:begin]
                        cursor = begin+size
                    assert next_state[cursor:] == state[cursor:]
                actual_f, expected_f = floats(output[:len(expected)]), floats(expected)
                errors = [a-b for a, b in zip(actual_f, expected_f)]
                max_reference = max(abs(x) for x in expected_f)
                error_sum = sum(x*x for x in errors)
                def relative_error(values):
                    norm = sum(x*x for x in values)
                    return math.sqrt(error_sum/norm) if norm else (0.0 if error_sum == 0 else None)
                relative_l2 = relative_error(expected_f)
                max_absolute = max(abs(x) for x in errors)
                input_f = floats(reference_input)
                reference_delta = [y-x for x, y in zip(input_f, expected_f)]
                delta_relative_l2 = relative_error(reference_delta)
                max_delta = max(abs(x) for x in reference_delta)
                numerical_ok = (relative_l2 is not None and delta_relative_l2 is not None
                                and relative_l2 <= RELATIVE_L2_LIMIT and delta_relative_l2 <= RELATIVE_L2_LIMIT
                                and max_absolute <= ABSOLUTE_FLOOR+MAX_RELATIVE_SCALE*max_reference
                                and max_absolute <= ABSOLUTE_FLOOR+MAX_RELATIVE_SCALE*max_delta)
                item = {'block': block, 'form': form, 'step': step, 'position': capture['position'],
                        'relative_l2': relative_l2, 'residual_delta_relative_l2': delta_relative_l2, 'max_absolute': max_absolute, 'max_reference': max_reference, 'max_reference_delta': max_delta,
                        'numerical_pass': numerical_ok, 'output_sha256': digest(output), 'state_sha256': digest(next_state)}
                record['comparisons'].append(item)
                (case/'actual.f32').write_bytes(output[:len(expected)])
                (case/'reference.f32').write_bytes(expected)
                if step == 0:
                    first = (output, next_state)
                if step == 'reset':
                    assert first == (output, next_state), (block, form, 'reset replay')
                key = (block, step)
                if form == 'loom':
                    variants[key] = (digest(output), digest(next_state))
                else:
                    assert variants[key] == (digest(output), digest(next_state)), (block, step, 'source/bytecode parity')
                state = next_state
                publish()
                # Retain small numerical outputs/hashes and logs, not copied banks.
                for path in capture_dir.glob('*.npy'):
                    path.unlink()
                for path in case.glob('arg*.npy'):
                    path.unlink()
                assert numerical_ok, item
                print(block, form, step, 'PASS', flush=True)
    for fd in banks:
        os.close(fd)
    record['passed'] = True
    publish()


if __name__ == '__main__':
    main()
