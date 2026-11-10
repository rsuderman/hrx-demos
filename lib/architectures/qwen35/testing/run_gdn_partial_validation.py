# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Actual complete GDN T3/N1 versus T1/N1, with poisoned inactive inputs.

No internal scratch is exposed or poisoned by this gate. Join the independent
production-clear/raw/recurrence/norm-gate witness for raw initialization proof.
Inactive residual rows inside static T may compute; only active output, full
persistent state and guards outside T have exact preservation requirements.
"""
import argparse
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

from run_block_validation import (
    ABSOLUTE_FLOOR, MAX_RELATIVE_SCALE, RELATIVE_L2_LIMIT,
    digest, floats, load, npy_header, save,
)


def cpu_error(actual, expected, input_bytes):
    actual, expected, inputs = floats(actual), floats(expected), floats(input_bytes)
    errors = [a-b for a, b in zip(actual, expected)]
    error_sum = sum(x*x for x in errors)

    def relative(values):
        norm = sum(x*x for x in values)
        return math.sqrt(error_sum/norm) if norm else (0.0 if error_sum == 0 else None)

    delta = [y-x for x, y in zip(inputs, expected)]
    output_l2, delta_l2 = relative(expected), relative(delta)
    maximum = max(abs(x) for x in errors)
    output_max, delta_max = max(abs(x) for x in expected), max(abs(x) for x in delta)
    passed = (output_l2 is not None and delta_l2 is not None
              and output_l2 <= RELATIVE_L2_LIMIT and delta_l2 <= RELATIVE_L2_LIMIT
              and maximum <= ABSOLUTE_FLOOR+MAX_RELATIVE_SCALE*output_max
              and maximum <= ABSOLUTE_FLOOR+MAX_RELATIVE_SCALE*delta_max)
    return {'relative_l2': output_l2, 'residual_delta_relative_l2': delta_l2,
            'max_absolute': maximum, 'max_reference': output_max,
            'max_reference_delta': delta_max, 'absolute_cpu_pass': passed}


def main(block='gdn0'):
    assert block in ('gdn0', 'attention3')
    attention = block == 'attention3'
    layer = '3' if attention else '0'
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('emitted', 'reference', 'gguf', 'reflector', 'runner', 'output', 'gpu-lock'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--retained08-baseline', type=Path,
                        help='Explicit preservation-only check; not an absolute CPU qualification')
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert sys.byteorder == 'little' and not os.environ.get('PYTHONPATH')
    assert not os.environ.get('PYTHONHOME')
    geometry = json.loads((args.emitted/'layout.json').read_text())
    reference = json.loads((args.reference/'capture.json').read_text())
    assert Path(reference['model']).resolve() == args.gguf.resolve()
    assert reference['n_gpu_layers'] == 0 and reference['kv_dtype'] == 'f16'
    hidden = geometry['hidden']
    assert hidden in (1024, 5120) and geometry['pool'] == 64
    if attention:
        assert hidden == 5120 and not args.retained08_baseline
    else:
        assert bool(args.retained08_baseline) == (hidden == 1024)
    assert [(x['step'], x['position'], x['token']) for x in reference['captures']] == [(0, 0, 100), (1, 1, 200), ('reset', 0, 100)]
    baseline = None
    if args.retained08_baseline:
        source = json.loads(args.retained08_baseline.read_text())
        assert source['geometry']['hidden'] == hidden
        baseline = next(x for x in source['comparisons'] if x['block'] == 'gdn0' and x['form'] == 'loom' and x['step'] == 0)
        assert not baseline['numerical_pass'], 'Preserve the known failed absolute08 qualification'
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PARAMETER_GGUF=str(args.gguf))
    record = {'passed': False,
              'scope': 'retained08 preservation only' if baseline else '27B complete '+block+' partial-active qualification',
              'internal_raw_poisoned': False,
              'caller_owned_attended_output_poisoned': attention,
              'raw_initialization_evidence': ('direct caller-owned attended padding plus separate width-clear witness' if attention
                                              else 'separate production clear/recurrence/norm-gate component witness'),
              'geometry': geometry, 'reference': reference,
              'criteria': {'output_relative_l2': RELATIVE_L2_LIMIT,
                           'residual_delta_relative_l2': RELATIVE_L2_LIMIT,
                           'absolute_floor': ABSOLUTE_FLOOR, 'relative_scale': MAX_RELATIVE_SCALE,
                           'zero_norm_rule': 'zero reference norm requires exactly zero error',
                           'active_and_full_state': 'bitwise T1 == T3 positive == T3 negative == source/BC'},
              'runs': [], 'comparisons': []}

    def publish():
        (args.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')

    def reference_bytes(capture, name):
        fact = capture['tensors'][name]
        assert fact['dtype'] == 'f32' and fact['shape'] == [hidden]
        path = (args.reference/fact['path']).resolve()
        assert path.parent == args.reference.resolve()
        data = path.read_bytes()
        assert len(data) == 4*hidden
        floats(data)
        return data

    state_origin = geometry['state_origin']
    payload_size = (2*geometry['pool']*geometry['plane_token_bytes'] if attention
                    else geometry['gdn_state_bytes'])
    attended_row_bytes = 4*geometry['attention_width'] if attention else 0
    initial_state = bytes([91])*state_origin + bytes(payload_size) + bytes([91])*256
    rows = bytearray(geometry['map_origin']+4)
    struct.pack_into('<4q', rows, 0, state_origin, state_origin, state_origin, 0)
    expected = {}
    payload_identity = None
    for form in ('loom', 'loombc'):
        module = args.emitted/('configured_blocks.'+form)
        for variant, static_rows, sign in [('t1', 1, 0), ('t3-positive', 3, 1), ('t3-negative', 3, -1)]:
            root = 'validated.qwen.'+block + ('_t3' if static_rows == 3 else '')
            work = args.output/(form+'-'+variant)
            work.mkdir(exist_ok=True)
            with (work/'parameters.tsv').open('w') as out, (work/'reflection.log').open('w') as err:
                subprocess.run([str(args.reflector), root, str(module)], env=env, stdout=out, stderr=err, check=True)
            roots = re.findall(r'ROOT (\d+) length=(\d+) alignment=(\d+)', (work/'reflection.log').read_text())
            assert len(roots) == 1 and roots[0][0] == '0'
            length = int(roots[0][1])
            fd = os.memfd_create('gdn-partial-actual-weights', os.MFD_CLOEXEC)
            header = npy_header(length)
            os.ftruncate(fd, len(header)+length)
            os.pwrite(fd, header, 0)
            payloads = []
            with args.gguf.open('rb', buffering=0) as source:
                for line in (work/'parameters.tsv').read_text().splitlines():
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
            assert sorted(x['name'] for x in payloads) == sorted(geometry['bindings'][layer])
            identity = sorted((x['name'], x['bytes'], x['source'], x['sha256']) for x in payloads)
            if payload_identity is None:
                payload_identity = identity
            assert payload_identity == identity
            record['runs'].append({'form': form, 'variant': variant, 'root': root,
                                   'module_sha256': digest(module.read_bytes()), 'payloads': payloads})
            state = initial_state
            replay = None
            for capture in reference['captures']:
                step = capture['step']
                case = work/str(step)
                case.mkdir(exist_ok=True)
                if step == 'reset':
                    state = initial_state
                active_input = reference_bytes(capture, 'l_out-2' if attention else 'model.input_embed')
                cpu_expected = reference_bytes(capture, 'l_out-3' if attention else 'l_out-0')
                residual = bytearray(active_input + struct.pack('<f', -31.)*((geometry['residual_bytes']-len(active_input))//4+64))
                if static_rows == 3:
                    for c in range(hidden):
                        struct.pack_into('<f', residual, 4*(hidden+c), sign*(16+(c%17)*8))
                        struct.pack_into('<f', residual, 4*(2*hidden+c), -sign*(256+(c%11)*32))
                control = bytearray(4192)
                struct.pack_into('<8i', control, 0, 1, 1, 0, 1, capture['position'], 0, 0, -1)
                attended_before = (struct.pack('<f', (sign or 1)*123.5)*(geometry['capture_bytes']//4+64)
                                   if attention else bytes([83])*geometry['capture_bytes'])
                buffers = [bytes(residual), state, bytes(control), attended_before, bytes(rows)]
                for i, data in enumerate(buffers, 1):
                    save(case/f'arg{i}.npy', data)
                declarations = ','.join(f'%a{i}:buffer' for i in range(6))
                tensor_types = ','.join([f'tensor<{length}xi8>']+[f'tensor<{len(x)}xi8>' for x in buffers])
                fixture = f'command.program.decl @{root}() launch({declarations})\nfunc.def @noop({declarations}) {{ func.return }}\ncheck.scenario @partial_gdn {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n'
                fixture += f'  %a0 = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>\n'
                for i, data in enumerate(buffers, 1):
                    fixture += f'  %a{i} = check.file.read.npy path("{case/f"arg{i}.npy"}") : tensor<{len(data)}xi8>\n'
                fixture += f'  check.compare<@{root},@noop>(%a0,%a1,%a2,%a3,%a4,%a5) : ({tensor_types}) -> () {{\n'
                for i, size in ((0, length), (3, len(control)), (5, len(rows))):
                    fixture += f'   check.expect.bitwise actual(%a{i}) expected(%a{i}) : tensor<{size}xi8>\n'
                fixture += '  }\n }\n check.return\n}\n'
                (case/'check.loom').write_text(fixture)
                capture_dir = case/'capture'
                capture_dir.mkdir(exist_ok=True)
                command = [str(args.runner), str(case/'check.loom'), '--library='+str(module), '--case=@partial_gdn', '--device=amdgpu']
                with args.gpu_lock.open('a') as lock, (case/'execution.log').open('w') as log:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = subprocess.run(command, env=dict(env, LOOM_FFN_CAPTURE_DIR=str(capture_dir)), stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
                assert result.returncode == 0, case/'execution.log'
                output = load(capture_dir/'product0_call0_after_arg1.npy')
                next_state = load(capture_dir/'product0_call0_after_arg2.npy')
                assert len(output) == len(residual) and len(next_state) == len(state)
                active_output = output[:4*hidden]
                floats(output[:static_rows*4*hidden])
                assert output[static_rows*4*hidden:] == residual[static_rows*4*hidden:], (variant, step, 'outside static T')
                assert next_state[:state_origin] == initial_state[:state_origin]
                assert next_state[state_origin+payload_size:] == initial_state[state_origin+payload_size:]
                active_attended = b''
                if attention:
                    attended = load(capture_dir/'product0_call0_after_arg4.npy')
                    assert len(attended) == len(attended_before)
                    floats(attended[:static_rows*attended_row_bytes])
                    assert attended[attended_row_bytes:static_rows*attended_row_bytes] == bytes((static_rows-1)*attended_row_bytes), (variant, step, 'attended inactive rows undefined')
                    assert attended[static_rows*attended_row_bytes:] == attended_before[static_rows*attended_row_bytes:], (variant, step, 'attended outside T guard')
                    active_attended = attended[:attended_row_bytes]
                    cursor = state_origin
                    for plane in (0, 1):
                        begin = state_origin+(plane*geometry['pool']+capture['position'])*geometry['plane_token_bytes']
                        assert next_state[cursor:begin] == state[cursor:begin], (variant, step, 'unselected cache bytes')
                        cursor = begin+geometry['plane_token_bytes']
                    assert next_state[cursor:] == state[cursor:], (variant, step, 'cache suffix')
                # The reviewed complete attention route calls the same H256 WMMA
                # kernel/template for T1/T3, both one tile and actual N1. The
                # coordinator must retain that route qualification for this exact
                # joined source; this does not assert cross-policy bitwise parity.
                observed = (active_output, next_state, active_attended)
                if form == 'loom' and variant == 't1':
                    expected[step] = observed
                assert expected[step] == observed, (form, variant, step, 'T1/T3 poison or source/BC independence')
                if step == 0:
                    replay = (output, next_state)
                if step == 'reset':
                    assert replay == (output, next_state), (form, variant, 'reset replay')
                errors = cpu_error(active_output, cpu_expected, active_input)
                item = dict(errors, form=form, variant=variant, static_T=static_rows, runtime_N=1,
                            step=step, position=capture['position'],
                            active_sha256=digest(active_output), output_sha256=digest(output), state_sha256=digest(next_state),
                            inactive_inside_T_contract='finite residual may compute; attended inactive rows must be zero' if attention else 'finite, may compute',
                            active_attended_sha256=digest(active_attended) if attention else None,
                            outside_T_and_state_guards_preserved=True)
                record['comparisons'].append(item)
                if baseline and step == 0 and variant == 't1':
                    assert item['output_sha256'] == baseline['output_sha256']
                    assert item['state_sha256'] == baseline['state_sha256']
                (case/'actual.f32').write_bytes(active_output)
                (case/'reference.f32').write_bytes(cpu_expected)
                state = next_state
                publish()
                for path in capture_dir.glob('*.npy'):
                    path.unlink()
                for path in case.glob('arg*.npy'):
                    path.unlink()
                if not baseline:
                    assert errors['absolute_cpu_pass'], item
                print(form, variant, step, 'PRESERVATION PASS' if baseline else 'PASS', flush=True)
            os.close(fd)
    record['absolute_cpu_qualified'] = False if baseline else all(x['absolute_cpu_pass'] for x in record['comparisons'])
    record['passed'] = True
    publish()


if __name__ == '__main__':
    main()
