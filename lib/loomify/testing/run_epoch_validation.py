# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
"""Finite actual-GGUF epoch equivalence/ownership gate, without a math oracle.

Consume emit_epoch_validation outputs at T8/C64. Compare B0/S0 and B2/S7
with the same logical inputs, position and persistent state; then continuation,
reset and no-output. Source/bytecode, full state, active residuals and output
slots must match bitwise. Run only in the coordinator's isolated venv/GPU lane.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import site
import struct
import subprocess
import sys

# Reuse the maintained bank runner's byte-preserving NPY/capture utilities.
helper_path = Path(__file__).resolve().parents[2] / 'architectures/qwen35/testing/run_block_validation.py'
spec = importlib.util.spec_from_file_location('epoch_capture_bytes', helper_path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('emitted', 'gguf', 'reflector', 'runner', 'output', 'gpu-lock'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert sys.byteorder == 'little'
    assert not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME')
    args.output.mkdir(parents=True, exist_ok=True)
    layout = json.loads((args.emitted / 'layout.json').read_text())
    assert layout['tokens'] == 8 and layout['context'] == 64
    family = layout['family']
    assert family in ('llama', 'qwen35')
    qwen = family == 'qwen35'
    hidden = layout['hidden']
    facts = {key: int(value) for key, value in re.findall(
        r'config.def @([\w.]+) = (\d+) :',
        (args.emitted / 'configuration.loom').read_text())}
    if qwen:
        state_begin = facts['qwen.model.gdn_slot_origin']
        kv_begin = facts['qwen.model.private_row_stride']
        state_bytes = kv_begin + 64 * facts['qwen.model.target_kv_token_bytes']
        residual_bytes = facts['qwen.model.hidden_capacity_bytes']
        metadata_bytes = 4192
        map_origin = facts['qwen.model.map_origin']
        eos = 0  # Qwen numerical publication returns the raw selected token.
        vocabulary = facts['qwen.model.vocabulary_size']
        plane_count = facts['qwen.model.target_kv_planes']
        plane_token_bytes = facts['qwen.model.kv_plane_token_bytes']
        gdn_end = state_begin + facts['qwen.model.gdn_state_bytes']
    else:
        state_begin = kv_begin = facts['llama.attention.cache_base']
        state_bytes = kv_begin + 4 * facts['llama.model.layer_count'] * 64 * facts['llama.model.kv_size']
        residual_bytes = 8 * hidden * 4
        metadata_bytes = 2064
        map_origin = facts['llama.attention.map_origin']
        eos = facts['llama.publication.end_turn_id']
        vocabulary = facts['llama.model.vocabulary_size']
        plane_count = 2 * facts['llama.model.layer_count']
        plane_token_bytes = 2 * facts['llama.model.kv_size']
        gdn_end = state_begin
    assert vocabulary > 400
    rows = bytearray([83]) * (map_origin + 4 + 64)
    struct.pack_into('<4q', rows, 0, state_begin, state_begin, kv_begin, 11)
    struct.pack_into('<i', rows, map_origin, 0)
    initial_state = bytes([91]) * state_begin + bytes(state_bytes - state_begin) + bytes([91]) * 256
    root = 'generated.qwen_epoch' if qwen else 'generated.llama_epoch'
    env = dict(os.environ, PYTHONNOUSERSITE='1', PARAMETER_GGUF=str(args.gguf))
    record = {'passed': False, 'layout': layout, 'criteria': 'bitwise equivalence/ownership; no numerical accuracy qualification',
              'runs': [], 'comparisons': []}
    prior_form = {}

    def publish():
        (args.output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')

    for form in ('loom', 'loombc'):
        module = args.emitted / ('epoch.' + form)
        work = args.output / form
        work.mkdir(exist_ok=True)
        with (work / 'parameters.tsv').open('w') as out, (work / 'reflection.log').open('w') as err:
            subprocess.run([str(args.reflector), root, str(module)], env=env, stdout=out, stderr=err, check=True)
        roots = re.findall(r'ROOT (\d+) length=(\d+) alignment=(\d+)', (work / 'reflection.log').read_text())
        assert len(roots) == 1 and roots[0][0] == '0'
        length = int(roots[0][1])
        fd = os.memfd_create('actual-epoch-immutable-bank', os.MFD_CLOEXEC)
        header = h.npy_header(length)
        os.ftruncate(fd, len(header) + length)
        os.pwrite(fd, header, 0)
        payloads = []
        with args.gguf.open('rb', buffering=0) as source:
            for line in (work / 'parameters.tsv').read_text().splitlines():
                name, offset, size, buffer, origin = line.split('\t')
                assert int(buffer) == 0
                source.seek(int(origin))
                remaining, destination = int(size), len(header) + int(offset)
                while remaining:
                    data = source.read(min(remaining, 1024 * 1024))
                    assert data and os.pwrite(fd, data, destination) == len(data)
                    remaining -= len(data)
                    destination += len(data)
                payloads.append({'name': name, 'bytes': int(size), 'source': int(origin)})
        assert payloads
        record['runs'].append({'form': form, 'module_sha256': h.digest(module.read_bytes()), 'bindings': payloads})

        def run(name, begin, slot, position, tokens, state, selected=True):
            case = work / name
            case.mkdir(exist_ok=True)
            epoch = bytearray([79]) * (metadata_bytes + 64)
            count = int(selected)
            struct.pack_into('<8i', epoch, 0, begin + len(tokens), 1, count,
                             len(tokens), position, begin, 0, slot if selected else -1)
            struct.pack_into('<i', epoch, 1292, begin + len(tokens) - 1)
            struct.pack_into('<i', epoch, 1292 + 7 * 4, 0)
            struct.pack_into('<2i', epoch, 1548, 0, 0)
            struct.pack_into('<i', epoch, 2060, eos)
            # Qwen eagerly embeds valid padding. Llama's stronger active guard
            # must avoid invalid IDs outside its descriptor's input span.
            ids = [0 if qwen else -1] * 8
            ids[begin:begin + len(tokens)] = tokens
            input_ids = struct.pack('<8i', *ids) + bytes([71]) * 64
            output_ids = struct.pack('<20i', *(-99 - 7 * i for i in range(20)))
            residual = struct.pack('<f', -31.) * (residual_bytes // 4 + 64)
            buffers = [residual, bytes(epoch), bytes(rows), state, input_ids, output_ids]
            for i, data in enumerate(buffers, 1):
                h.save(case / f'arg{i}.npy', data)
            declarations = ','.join(f'%a{i}:buffer' for i in range(7))
            tensors = ','.join([f'tensor<{length}xi8>'] + [f'tensor<{len(x)}xi8>' for x in buffers])
            fixture = f'command.program.decl @{root}() launch({declarations})\nfunc.def @noop({declarations}) {{ func.return }}\ncheck.scenario @actual_epoch {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n'
            fixture += f'  %a0 = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>\n'
            for i, data in enumerate(buffers, 1):
                fixture += f'  %a{i} = check.file.read.npy path("{case / f"arg{i}.npy"}") : tensor<{len(data)}xi8>\n'
            fixture += f'  check.compare<@{root},@noop>(%a0,%a1,%a2,%a3,%a4,%a5,%a6) : ({tensors}) -> () {{\n'
            for i in (0, 2, 3, 5):
                size = length if i == 0 else len(buffers[i - 1])
                fixture += f'   check.expect.bitwise actual(%a{i}) expected(%a{i}) : tensor<{size}xi8>\n'
            fixture += '  }\n }\n check.return\n}\n'
            (case / 'check.loom').write_text(fixture)
            captures = case / 'capture'
            captures.mkdir(exist_ok=True)
            command = [str(args.runner), str(case / 'check.loom'), '--library=' + str(module), '--case=@actual_epoch', '--device=amdgpu']
            with args.gpu_lock.open('a') as lock, (case / 'execution.log').open('w') as log:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(command, env=dict(env, LOOM_FFN_CAPTURE_DIR=str(captures)),
                                        stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
            assert result.returncode == 0, case / 'execution.log'
            actual = {i: h.load(captures / f'product0_call0_after_arg{i}.npy') for i in range(1, 7)}
            assert actual[1][residual_bytes:] == residual[residual_bytes:]
            assert actual[4][:state_begin] == state[:state_begin]
            assert actual[4][state_bytes:] == state[state_bytes:]
            assert actual[4] != state, (form, name, 'persistent computation must execute')
            permitted = [(state_begin, gdn_end)] if qwen else []
            permitted += [(kv_begin + (plane * 64 + position) * plane_token_bytes,
                           kv_begin + (plane * 64 + position + len(tokens)) * plane_token_bytes)
                          for plane in range(plane_count)]
            cursor = 0
            for low, high in sorted(permitted):
                assert actual[4][cursor:low] == state[cursor:low], (form, name, 'unrelated state')
                cursor = high
            assert actual[4][cursor:] == state[cursor:], (form, name, 'unrelated state tail')
            assert actual[2] == buffers[1] and actual[3] == buffers[2] and actual[5] == buffers[4]
            out = bytearray(output_ids)
            if selected:
                out[slot * 4:slot * 4 + 4] = actual[6][slot * 4:slot * 4 + 4]
            assert actual[6] == bytes(out), (form, name, 'only selected physical output may change')
            active_begin = begin if qwen else 0
            active = actual[1][active_begin * hidden * 4:(active_begin + len(tokens)) * hidden * 4]
            values = h.floats(active)
            assert all(math.isfinite(value) for value in values), (form, name, "nonfinite active residual")
            assert any(x != 0. for x in values) and active != residual[:len(active)]
            selected_id = actual[6][slot * 4:slot * 4 + 4] if selected else None
            if selected:
                assert 0 <= struct.unpack('<i', selected_id)[0] < vocabulary
            proof = (h.digest(active), h.digest(actual[4]), selected_id.hex() if selected_id else None)
            if form == 'loom':
                prior_form[name] = proof
            else:
                assert prior_form[name] == proof, (name, 'SOURCE/BC')
            record['comparisons'].append({'form': form, 'case': name, 'position': position, 'begin': begin,
                                          'length': len(tokens), 'slot': slot if selected else -1,
                                          'active_sha256': proof[0], 'full_state_sha256': proof[1], 'selected_i32_bytes': proof[2]})
            publish()
            for path in captures.glob('*.npy'):
                path.unlink()
            for path in case.glob('arg*.npy'):
                path.unlink()
            return proof, actual[4]

        canonical, canonical_state = run('canonical', 0, 0, 11, [100, 200, 300], initial_state)
        offset, offset_state = run('offset', 2, 7, 11, [100, 200, 300], initial_state)
        assert canonical == offset, (form, 'B0/S0 vs B2/S7')
        continuation, _ = run('canonical_continuation', 0, 0, 14, [400], canonical_state)
        offset_continuation, _ = run('offset_continuation', 2, 7, 14, [400], offset_state)
        assert continuation == offset_continuation, (form, 'dependent continuation')
        reset, _ = run('reset', 2, 7, 11, [100, 200, 300], initial_state)
        assert reset == offset, (form, 'reset replay')
        no_output, _ = run('no_output', 2, 7, 11, [100, 200, 300], initial_state, False)
        assert no_output[:2] == offset[:2], (form, 'unselected computation/state')
        os.close(fd)
    record['passed'] = True
    publish()


if __name__ == '__main__':
    main()
