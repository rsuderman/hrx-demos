# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
"""Manual real-GGUF gate for maintained decoder-layer state and ABI invariants.

Run under a task-owned venv, with the native emitter already run. All products,
state and evidence go to --output. Neither weights nor model source are changed.
The coordinator owns execution and GPU scheduling; no downloads are performed.
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
import time


def header(kind, count):
    text = repr({'descr': kind, 'fortran_order': False, 'shape': (count,)})
    text += ' ' * ((64 - (10 + len(text) + 1) % 64) % 64) + '\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(text)) + text.encode()


def save(path, kind, data, width=1):
    path.write_bytes(header(kind, len(data) // width) + data)


def load(path):
    data = path.read_bytes()
    size = 2 if data[6] == 1 else 4
    count = int.from_bytes(data[8:8 + size], 'little')
    ast.literal_eval(data[8 + size:8 + size + count].decode())
    return data[8 + size + count:]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('emitted', 'gguf', 'reflector', 'runner', 'output', 'gpu-lock'):
        parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert sys.byteorder == 'little'
    assert not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME')
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((args.emitted / 'layout.json').read_text())
    env = dict(os.environ, PYTHONNOUSERSITE='1', PARAMETER_GGUF=str(args.gguf))
    modules = {'configured': args.emitted / 'configured_layer.loombc'}
    record = {'modules': {name: {'path': str(path), 'sha256': sha(path.read_bytes())}
                          for name, path in modules.items()},
              'geometry': geometry, 'runs': [], 'comparisons': []}
    banks = {}

    def publish():
        (args.output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')

    def bank(kind, root):
        key = (kind, root)
        if key in banks:
            return banks[key]
        prefix = args.output / (kind + '-' + root)
        command = [str(args.reflector), root, str(modules[kind])]
        with Path(str(prefix) + '.parameters.tsv').open('w') as out, \
                Path(str(prefix) + '.reflection.log').open('w') as err:
            subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
        roots = re.findall(r'ROOT (\d+) length=(\d+) alignment=(\d+)',
                           Path(str(prefix) + '.reflection.log').read_text())
        assert len(roots) == 1 and roots[0][0] == '0', roots
        length = int(roots[0][1])
        fd = os.memfd_create('llama-real-layer-parameters', os.MFD_CLOEXEC)
        preamble = header('|i1', length)
        os.ftruncate(fd, len(preamble) + length)
        os.pwrite(fd, preamble, 0)
        payloads = []
        with args.gguf.open('rb', buffering=0) as checkpoint:
            for line in Path(str(prefix) + '.parameters.tsv').read_text().splitlines():
                name, offset, size, buffer, source = line.split('\t')
                assert int(buffer) == 0
                checkpoint.seek(int(source))
                remaining, destination = int(size), len(preamble) + int(offset)
                digest = hashlib.sha256()
                while remaining:
                    chunk = checkpoint.read(min(remaining, 1024 * 1024))
                    assert chunk
                    digest.update(chunk)
                    assert os.pwrite(fd, chunk, destination) == len(chunk)
                    remaining -= len(chunk)
                    destination += len(chunk)
                payloads.append({'name': name, 'offset': int(offset),
                                 'size': int(size), 'source': int(source),
                                 'sha256': digest.hexdigest()})
        assert len(payloads) == 10, payloads
        # Exact reflected roots can order bindings differently. Both artifacts
        # must still resolve the same ten original nonzero tensor byte ranges.
        banks[key] = (fd, length, payloads)
        return banks[key]

    residual = array.array('f', (0.01 + (i % 1024) * 0.000001
                               for i in range(17 * 3072 + 64))).tobytes()
    initial_cache = bytes([0, 52]) * ((geometry['cache_bytes'] + 256) // 2)
    initial_map = bytearray([91]) * (geometry['map_bytes'] + 64)
    # A permutation makes page63->64 prove mapping rather than identity offsets.
    block_map = [3, 0, 7, 1, 6, 2, 5, 4]
    for page, block in enumerate(block_map):
        struct.pack_into('<i', initial_map, geometry['map_origin'] + page * 4, block)
    map_bytes = bytes(initial_map)
    cases = [('page-prefill', 0, 3, 61, False),
             ('page-decode', 0, 1, 64, True),
             ('reset', 0, 3, 61, False),
             ('q3-pos255', 0, 3, 253, False),
             ('q4-pos255', 0, 4, 252, False),
             ('q3-pos256', 0, 3, 254, False),
             ('q4-pos256', 0, 4, 253, False),
             ('q16-pos255', 1, 16, 240, False),
             ('q17-pos255', 1, 17, 239, False),
             ('q16-pos256', 1, 16, 241, False),
             ('q17-pos256', 1, 17, 240, False)]
    continuation = {}
    first_result = None
    for label, layer, active, base, decode in cases:
        root = f'validated.layer{layer}.' + ('decode' if decode else 'prefill')
        control = struct.pack('<3i', base if decode else active,
                              7 if decode else base, 0x1234)
        outputs = {}
        for kind in modules:
            fd, length, payloads = bank(kind, root)
            work = args.output / (label + '-' + kind)
            work.mkdir(exist_ok=True)
            before_cache = continuation[kind] if decode else initial_cache
            for name, data in [('residual', residual), ('control', control),
                               ('map', map_bytes), ('cache', before_cache)]:
                save(work / (name + '.npy'), '|i1', data)
            fixture = work / 'fixture.loom'
            fixture.write_text(f'''command.decl @{root}() launch(%p:buffer,%r:buffer,%c:buffer,%m:buffer,%k:buffer)
func.def @noop(%p:buffer,%r:buffer,%c:buffer,%m:buffer,%k:buffer) {{ func.return }}
check.scenario @layer_validation {{
 check.trial[1](%trial:index,%entropy:check.entropy) {{
  %p = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>
  %r = check.file.read.npy path("{work/'residual.npy'}") : tensor<{len(residual)}xi8>
  %c = check.file.read.npy path("{work/'control.npy'}") : tensor<12xi8>
  %m = check.file.read.npy path("{work/'map.npy'}") : tensor<{len(map_bytes)}xi8>
  %k = check.file.read.npy path("{work/'cache.npy'}") : tensor<{len(before_cache)}xi8>
  check.compare<@{root},@noop>(%p,%r,%c,%m,%k) : (tensor<{length}xi8>,tensor<{len(residual)}xi8>,tensor<12xi8>,tensor<{len(map_bytes)}xi8>,tensor<{len(before_cache)}xi8>) -> () {{
   check.expect.bitwise actual(%p) expected(%p) : tensor<{length}xi8>
   check.expect.bitwise actual(%c) expected(%c) : tensor<12xi8>
   check.expect.bitwise actual(%m) expected(%m) : tensor<{len(map_bytes)}xi8>
  }}
 }}
 check.return
}}
''')
            capture = work / 'capture'
            capture.mkdir(exist_ok=True)
            runenv = dict(env, LOOM_FFN_CAPTURE_DIR=str(capture))
            command = [str(args.runner), str(fixture), '--library=' + str(modules[kind]),
                       '--case=@layer_validation', '--device=amdgpu']
            start = time.monotonic()
            with args.gpu_lock.open('a') as lock, (work / 'execution.log').open('w') as log:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(command, env=runenv, stdout=log,
                                        stderr=subprocess.STDOUT, pass_fds=(fd,))
            record['runs'].append({'case': label, 'kind': kind, 'command': command,
                                   'exit': result.returncode, 'seconds': time.monotonic()-start,
                                   'payloads': payloads})
            publish()
            assert result.returncode == 0, (label, kind, work / 'execution.log')
            output_residual = load(capture / 'product0_call0_after_arg1.npy')
            output_cache = load(capture / 'product0_call0_after_arg4.npy')
            assert output_residual[active * 3072 * 4:] == residual[active * 3072 * 4:], label
            values = array.array('f')
            values.frombytes(output_residual[:active * 3072 * 4])
            assert all(math.isfinite(value) for value in values), label
            assert output_residual[:active * 3072 * 4] != residual[:active * 3072 * 4], label
            permitted = []
            stride = geometry['pool'] * 1024 * 2
            for plane in (2 * layer, 2 * layer + 1):
                for position in range(base, base + active):
                    physical = block_map[position // 64] * 64 + position % 64
                    origin = geometry['cache_base'] + plane * stride + physical * 2048
                    permitted.append((origin, origin + 2048))
            cursor = 0
            for begin, end in sorted(permitted):
                assert output_cache[cursor:begin] == before_cache[cursor:begin], (label, kind, cursor, begin)
                cursor = end
            assert output_cache[cursor:] == before_cache[cursor:], (label, kind, 'suffix')
            if label == 'page-prefill':
                continuation[kind] = output_cache
            outputs[kind] = (output_residual, output_cache)
            # Keep logs, fixtures and hashes; large transient captures are not evidence.
            for path in capture.glob('*.npy'):
                path.unlink()
            for name in ('residual', 'control', 'map', 'cache'):
                (work / (name + '.npy')).unlink()
        if label == 'page-prefill':
            first_result = tuple(sha(value) for value in outputs['configured'])
        if label == 'reset':
            assert first_result == tuple(sha(value) for value in outputs['configured'])
        record['comparisons'].append({'case': label, 'layer': layer, 'active': active,
            'base': base, 'decode': decode, 'coverage': 'state/ABI; independent arithmetic remains in operation suites',
            'inactive_residual_other_planes_prefix_suffix_preserved': True,
            'finite_nontrivial_output': True,
            'residual_sha256': sha(outputs['configured'][0]),
            'cache_sha256': sha(outputs['configured'][1])})
        publish()
        print(label, 'PASS', flush=True)
    for fd, _, _ in banks.values():
        os.close(fd)
    record['passed'] = True
    publish()


if __name__ == '__main__':
    main()
