# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
"""Finite actual-GGUF epoch equivalence/ownership gate, without a math oracle.

Consume emit_epoch_validation outputs at T8/C64 (Gemma: C576). Compare B0/S0 and B2/S7
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


def gemma_window_witness(module, windows):
    """Check this finite materialization's actual all-layer window consumer.

    This is a representation witness for the maintained constructor, not an IR
    interpreter or a substitute for the separate independent window math oracle.
    """
    text = module.read_text()
    assert len(re.findall(r'kernel.launch @gemma3_attention_rows\[', text)) == len(windows)
    match = re.search(r'^kernel.def @gemma3_attention_rows\b.*?(?=^kernel.def |\Z)', text, re.M | re.S)
    assert match, 'missing actual Gemma attention consumer'
    body = match.group()
    # Native LINK folds the identical first five arms into the local default.
    # Remaining explicit arms retain layer identity and their selected window.
    arms = {int(i): int(value) for i, value in re.findall(
        r'%c(\d+)_0 = index.constant (\d+) : index', body)}
    assert arms == {i: windows[i] for i in range(5, 25)}, arms
    for i in range(5, 25):
        assert f'%l{i} = index.constant {i} : index' in body
        assert f'%is{i} = index.cmp eq, %layer, %l{i} : index' in body
    assert '%v5_0 = index.constant 512 : index' in body
    assert '%windowed = index.cmp ne, %v25_0, %c0 : index' in body
    assert '%window_before = index.cmp ugt, %context_count, %v25_0 : index' in body
    assert '%key_begin = scf.if %trim -> (index)' in body
    return {'module_sha256': h.digest(module.read_bytes()),
            'attention_kernel_sha256': h.digest(body.encode()),
            'actual_attention_launches': len(windows), 'effective_windows': windows}


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
    family = layout['family']
    assert family in ('llama', 'qwen35', 'gemma3')
    qwen = family == 'qwen35'
    gemma = family == 'gemma3'
    context = layout['context']
    assert layout['tokens'] == 8 and context == (576 if gemma else 64)
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
        prefix = 'gemma3' if gemma else 'llama'
        state_begin = kv_begin = facts[prefix + '.attention.cache_base']
        state_bytes = kv_begin + 4 * facts[prefix + '.model.layer_count'] * context * facts[prefix + '.model.kv_size']
        residual_bytes = 8 * hidden * 4
        metadata_bytes = 2064
        map_origin = facts[prefix + '.attention.map_origin']
        eos = facts[prefix + '.publication.end_turn_id']
        vocabulary = facts[prefix + '.model.vocabulary_size']
        plane_count = 2 * facts[prefix + '.model.layer_count']
        plane_token_bytes = 2 * facts[prefix + '.model.kv_size']
        gdn_end = state_begin
    assert vocabulary > 400
    rows = bytearray([83]) * (map_origin + 4 + 64)
    struct.pack_into('<4q', rows, 0, state_begin, state_begin, kv_begin, 11)
    if gemma:
        for page in range(context // 64):
            struct.pack_into('<i', rows, map_origin + page * 4, page)
    else:
        struct.pack_into('<i', rows, map_origin, 0)
    initial_state = bytes([91]) * state_begin + bytes(state_bytes - state_begin) + bytes([91]) * 256
    root = ('generated.qwen_epoch' if qwen else
            'generated.gemma3_epoch' if gemma else 'generated.llama_epoch')
    env = dict(os.environ, PYTHONNOUSERSITE='1', PARAMETER_GGUF=str(args.gguf))
    record = {'passed': False, 'layout': layout, 'criteria': 'bitwise equivalence/ownership; no numerical accuracy qualification',
              'runs': [], 'comparisons': []}
    prior_form = {}
    if gemma:
        assert layout['layer_windows'] == [512 if (i + 1) % 6 else 0 for i in range(26)]
        assert layout['layer_rope_bases'] == [10000 if window else 1000000 for window in layout['layer_windows']]
        assert plane_count == 52 and plane_token_bytes == 512
        record['window_validation'] = {
            'native_route': gemma_window_witness(args.emitted / 'epoch.loom', layout['layer_windows']),
            'configuration_sha256': h.digest((args.emitted / 'configuration.loom').read_bytes()),
            'layer_windows': layout['layer_windows'],
            'layer_rope_bases': layout['layer_rope_bases'],
            'numerical_oracle': 'separate shared attention window oracle; this gate proves full-model routing and ownership',
            'criteria': 'P511/N3 then dependent P514/N1; excluded local KV invariance and included lower-edge sensitivity'}

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

        observations = {}

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
            permitted += [(kv_begin + (plane * context + position) * plane_token_bytes,
                           kv_begin + (plane * context + position + len(tokens)) * plane_token_bytes)
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
            if gemma:
                active_end = (active_begin + len(tokens)) * hidden * 4
                assert actual[1][active_end:residual_bytes] == residual[active_end:residual_bytes], (form, name, 'inactive residual')
            selected_id = actual[6][slot * 4:slot * 4 + 4] if selected else None
            if selected:
                assert 0 <= struct.unpack('<i', selected_id)[0] < vocabulary
            proof = (h.digest(active), h.digest(actual[4]), selected_id.hex() if selected_id else None)
            if gemma:
                observations[name] = (h.digest(active), selected_id, tuple(
                    h.digest(actual[4][kv_begin + (plane * context + position) * plane_token_bytes:
                                       kv_begin + (plane * context + position + len(tokens)) * plane_token_bytes])
                    for plane in range(plane_count)))
            if form == 'loom':
                prior_form[name] = proof
            else:
                assert prior_form[name] == proof, (name, 'SOURCE/BC')
            comparison = {'form': form, 'case': name, 'position': position, 'begin': begin,
                          'length': len(tokens), 'slot': slot if selected else -1,
                          'active_sha256': proof[0], 'full_state_sha256': proof[1], 'selected_i32_bytes': proof[2]}
            if gemma:
                comparison['input_state_sha256'] = h.digest(state)
                comparison['written_kv_sha256'] = observations[name][2]
            record['comparisons'].append(comparison)
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
        if gemma:
            # Finite, fully defined prior F16 K/V. This is a legal synthetic cache,
            # not a claim that the prefix was produced by a reference tokenizer/model.
            history = bytearray(initial_state)
            width = plane_token_bytes // 2
            for plane in range(plane_count):
                for position in range(511):
                    values = [(((plane * 7 + position * 3 + lane) % 31) - 15) / 64.
                              for lane in range(width)]
                    offset = kv_begin + (plane * context + position) * plane_token_bytes
                    history[offset:offset + plane_token_bytes] = struct.pack('<' + 'e' * width, *values)
            history = bytes(history)
            window, window_state = run('window_prefill', 0, 0, 511, [100, 200, 300], history)
            window_offset, window_offset_state = run('window_offset', 2, 7, 511, [100, 200, 300], history)
            assert window == window_offset, (form, 'window crossing B/S equivalence')
            window_decode, _ = run('window_decode', 0, 0, 514, [400], window_state)
            window_decode_offset, _ = run('window_offset_decode', 2, 7, 514, [400], window_offset_state)
            assert window_decode == window_decode_offset, (form, 'window dependent decode')
            excluded = bytearray(window_state)
            for layer, window_size in enumerate(layout['layer_windows']):
                if not window_size:
                    continue
                for plane in (2 * layer, 2 * layer + 1):
                    for position in range(3):  # P514 includes [3, 514], not [0, 2].
                        offset = kv_begin + (plane * context + position) * plane_token_bytes
                        excluded[offset:offset + plane_token_bytes] = struct.pack('<' + 'e' * width, *([4.0] * width))
            assert bytes(excluded) != window_state
            run('window_excluded_local_prefix', 0, 0, 514, [400], bytes(excluded))
            assert observations['window_excluded_local_prefix'] == observations['window_decode'], (form, 'excluded local prefix changed computation')
            # Position3 is the included lower edge. Change only the first local
            # layer's V row so full-model sensitivity has one unambiguous cause.
            included = bytearray(window_state)
            offset = kv_begin + (context + 3) * plane_token_bytes
            included[offset:offset + plane_token_bytes] = struct.pack('<' + 'e' * width, *([8.0] * width))
            run('window_included_local_edge', 0, 0, 514, [400], bytes(included))
            assert observations['window_included_local_edge'][0] != observations['window_decode'][0], (form, 'included local lower edge had no effect')
            global_prefix = bytearray(window_state)
            first_global = layout['layer_windows'].index(0)
            plane = 2 * first_global + 1
            for position in range(3):
                offset = kv_begin + (plane * context + position) * plane_token_bytes
                global_prefix[offset:offset + plane_token_bytes] = struct.pack('<' + 'e' * width, *([8.0] * width))
            run('window_included_global_prefix', 0, 0, 514, [400], bytes(global_prefix))
            assert observations['window_included_global_prefix'][0] != observations['window_decode'][0], (form, 'global attention incorrectly discarded prefix')
            record['window_validation'].setdefault('metamorphic_checks', []).append({
                'form': form, 'excluded_local_prefix_unchanged': True,
                'included_local_lower_edge_sensitive': True, 'global_prefix_sensitive': True})
            publish()
        os.close(fd)
    record['passed'] = True
    publish()


if __name__ == '__main__':
    main()
