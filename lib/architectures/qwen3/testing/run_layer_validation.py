# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Manual actual-GGUF composed numerical/state gate; run in an isolated venv.

Uses the maintained independent staged-half golden helpers, current native
parameter reflection and capture runner. Model inputs remain immutable. This is
one actual first-layer engineering fixture, not full-model fidelity evidence.
"""

import argparse
import array
import ast
import fcntl
import hashlib
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
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def npy_header(size):
    text = repr({"descr": "|i1", "fortran_order": False, "shape": (size,)})
    text += " " * ((64 - (10 + len(text) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(text)) + text.encode()


def save(path, data):
    path.write_bytes(npy_header(len(data)) + data)


def load(path):
    data = path.read_bytes()
    width = 2 if data[6] == 1 else 4
    size = int.from_bytes(data[8 : 8 + width], "little")
    ast.literal_eval(data[8 + width : 8 + width + size].decode())
    return data[8 + width + size :]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def floats(data):
    values = array.array("f")
    values.frombytes(data)
    return values


def close(actual, expected, name):
    assert len(actual) == len(expected)
    assert all(math.isfinite(x) for x in actual)
    assert all(math.isfinite(x) for x in expected)
    errors = [float(a) - e for a, e in zip(actual, expected)]
    relative = math.sqrt(math.fsum(x * x for x in errors)) / max(
        math.sqrt(math.fsum(float(x) * x for x in expected)), 1e-12
    )
    maximum = max(map(abs, errors), default=0.0)
    result = {
        "name": name,
        "values": len(actual),
        "relative_l2": relative,
        "max_abs": maximum,
        "relative_l2_limit": 0.02,
        "max_abs_limit": 0.03,
    }
    return result, relative <= 0.02 and maximum <= 0.03


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("emitted", "gguf", "reflector", "runner", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    assert sys.byteorder == "little"
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[4]
    helper = source / "lib/programs/projection/testing/matrix"
    fixture = module("fixtures", helper / "fixtures.py")
    golden = module("qwen3_independent_goldens", helper / "goldens.py")
    f32, f16 = fixture.f32, fixture.f16
    geometry = json.loads((args.emitted / "layout.json").read_text())
    t, e, h = (geometry[k] for k in ("rows", "hidden", "head"))
    qh, kh = geometry["query_heads"], geometry["kv_heads"]
    kv = kh * h
    assert (t, e, h, qh, kh) == (17, 1024, 128, 16, 8)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PARAMETER_GGUF=str(args.gguf))
    record = {
        "status": "running",
        "geometry": geometry,
        "oracle": "independent raw GGUF staged-F16 decode, F64 dot/softmax with explicit F32/F16 boundaries",
        "helpers": {
            str(p.relative_to(source)): digest(p.read_bytes())
            for p in (helper / "fixtures.py", helper / "goldens.py")
        },
        "budget": {"relative_l2": 0.02, "max_abs": 0.03},
        "runs": [],
        "comparisons": [],
        "independent_goldens": {},
    }

    def publish():
        (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")

    publish()
    weights, tensors = {}, {}
    with args.gguf.open("rb", buffering=0) as checkpoint:
        for tensor in geometry["tensors"]:
            name = tensor["name"]
            checkpoint.seek(tensor["offset"])
            weights[name] = checkpoint.read(tensor["bytes"])
            assert len(weights[name]) == tensor["bytes"]
            tensors[name] = tensor
    decoded = {}

    def matrix(inputs, suffix):
        name = "blk.0." + suffix + ".weight"
        fact = tensors[name]
        if name not in decoded:
            print("decode", name, flush=True)
            decoded[name] = [
                array.array("f", row)
                for row in golden.decode_staged(
                    weights[name], fact["tag"], fact["input"], fact["output"]
                )
            ]
        prepared = [golden.prepare(row) for row in inputs]
        return [
            [
                f32(
                    f32(math.fsum(float(x) * w for x, w in zip(values, weight_row)))
                    * inverse
                )
                for weight_row in decoded[name]
            ]
            for values, _, inverse in prepared
        ]

    def gamma(suffix):
        return floats(weights["blk.0." + suffix + ".weight"])

    def add(a, b):
        return [
            [f32(x + y) for x, y in zip(a_row, b_row)] for a_row, b_row in zip(a, b)
        ]

    def ffn(inputs):
        normalized = [
            golden.rms(row, gamma("ffn_norm"), geometry["epsilon"]) for row in inputs
        ]
        gate, up = matrix(normalized, "ffn_gate"), matrix(normalized, "ffn_up")
        gated = [
            [f32(f32(golden.silu(x)) * y) for x, y in zip(g, u)]
            for g, u in zip(gate, up)
        ]
        return add(inputs, matrix(gated, "ffn_down"))

    block_map = [1, 0]

    def physical(position):
        return block_map[position // 64] * 64 + position % 64

    stride = geometry["pool"] * kv * 2
    cache_base = geometry["cache_base"]

    def cache_offset(plane, position):
        return cache_base + plane * stride + physical(position) * kv * 2

    def attention(inputs, before, base):
        normalized = [
            golden.rms(row, gamma("attn_norm"), geometry["epsilon"]) for row in inputs
        ]
        q, k, v = (
            matrix(normalized, suffix) for suffix in ("attn_q", "attn_k", "attn_v")
        )
        rotated_q, rotated_k = [], []
        for rows, heads, suffix, output in [
            (q, qh, "attn_q_norm", rotated_q),
            (k, kh, "attn_k_norm", rotated_k),
        ]:
            for row, values in enumerate(rows):
                prepared = []
                for head in range(heads):
                    prepared += golden.neox(
                        golden.rms(
                            values[head * h : (head + 1) * h],
                            gamma(suffix),
                            geometry["epsilon"],
                        ),
                        base + row,
                        geometry["rope_log_step"],
                    )
                output.append(prepared)
        cache = bytearray(before)
        for row, (keys, values) in enumerate(zip(rotated_k, v)):
            for plane, source_row in [(0, keys), (1, values)]:
                origin = cache_offset(plane, base + row)
                struct.pack_into(
                    "<" + "e" * kv, cache, origin, *(f16(x) for x in source_row)
                )
        attended = []
        for row, query in enumerate(rotated_q):
            result = []
            for head in range(qh):
                key_head = head // (qh // kh)
                key_rows, value_rows, scores = [], [], []
                for position in range(base + row + 1):
                    key_row = struct.unpack_from(
                        "<" + "e" * h,
                        cache,
                        cache_offset(0, position) + key_head * h * 2,
                    )
                    value_row = struct.unpack_from(
                        "<" + "e" * h,
                        cache,
                        cache_offset(1, position) + key_head * h * 2,
                    )
                    key_rows.append(key_row)
                    value_rows.append(value_row)
                    scores.append(
                        f32(
                            math.fsum(
                                float(x) * y
                                for x, y in zip(
                                    query[head * h : (head + 1) * h], key_row
                                )
                            )
                            * geometry["attention_scale"]
                        )
                    )
                peak = max(scores)
                probability = [math.exp(x - peak) for x in scores]
                denominator = math.fsum(probability)
                result += [
                    f32(
                        math.fsum(
                            p * value_row[column]
                            for p, value_row in zip(probability, value_rows)
                        )
                        / denominator
                    )
                    for column in range(h)
                ]
            attended.append(result)
        return add(inputs, matrix(attended, "attn_output")), bytes(cache)

    inputs = [
        [
            f32(0.02 * math.sin(i * 0.17 + row * 0.3) + 0.003 * ((i + row) % 5 - 2))
            for i in range(e)
        ]
        for row in range(2)
    ]
    initial_cache = struct.pack("<e", 0.125) * ((geometry["cache_bytes"] + 256) // 2)
    print("independent FFN", flush=True)
    expected_ffn = ffn(inputs)
    print("independent attention/layer page63", flush=True)
    expected_attention, expected_cache = attention(inputs, initial_cache, 63)
    expected_layer = ffn(expected_attention)
    decode_inputs = [inputs[1]]
    print("independent dependent decode page65", flush=True)
    decode_attention, decode_cache = attention(decode_inputs, expected_cache, 65)
    expected_decode = ffn(decode_attention)
    expected_by_case = {
        "ffn": (expected_ffn, initial_cache),
        "attention": (expected_attention, expected_cache),
        "layer": (expected_layer, expected_cache),
        "poison": (expected_layer, expected_cache),
        "decode": (expected_decode, decode_cache),
    }
    for name, (rows, cache) in expected_by_case.items():
        data = array.array("f", (x for row in rows for x in row)).tobytes()
        save(args.output / (name + "-expected-residual.npy"), data)
        record["independent_goldens"][name] = {
            "residual_sha256": digest(data),
            "cache_sha256": digest(cache),
        }
    map_data = bytearray([91]) * (geometry["map_bytes"] + 64)
    for page, block in enumerate(block_map):
        struct.pack_into("<i", map_data, geometry["map_origin"] + page * 4, block)
    map_data = bytes(map_data)
    modules = {
        "source": args.emitted / "configured_layer.loom",
        "bc": args.emitted / "configured_layer.loombc",
    }
    record["modules"] = {
        name: digest(path.read_bytes()) for name, path in modules.items()
    }
    banks = {}
    source_outputs = {}

    def bank(kind, root):
        key = kind, root
        if key in banks:
            return banks[key]
        prefix = args.output / (kind + "-" + root)
        with (
            Path(str(prefix) + ".tsv").open("w") as out,
            Path(str(prefix) + ".log").open("w") as log,
        ):
            subprocess.run(
                [str(args.reflector), root, str(modules[kind])],
                env=env,
                stdout=out,
                stderr=log,
                check=True,
            )
        roots = re.findall(
            r"ROOT (\d+) length=(\d+) alignment=(\d+)",
            Path(str(prefix) + ".log").read_text(),
        )
        assert len(roots) == 1 and roots[0][0] == "0", roots
        size = int(roots[0][1])
        fd = os.memfd_create("qwen3-independent-layer-weights", os.MFD_CLOEXEC)
        header = npy_header(size)
        os.ftruncate(fd, len(header) + size)
        os.pwrite(fd, header, 0)
        bindings = []
        for line in Path(str(prefix) + ".tsv").read_text().splitlines():
            name, offset, length, buffer, source_offset = line.split("\t")
            assert int(buffer) == 0 and name in weights
            assert int(source_offset) == tensors[name]["offset"]
            assert int(length) == len(weights[name])
            assert os.pwrite(fd, weights[name], len(header) + int(offset)) == int(
                length
            )
            bindings.append(
                {
                    "name": name,
                    "offset": int(offset),
                    "bytes": int(length),
                    "source": int(source_offset),
                    "sha256": digest(weights[name]),
                }
            )
        assert (
            len(bindings)
            == {"ffn0": 4, "attention0": 7, "layer0": 11}[root.rsplit(".", 1)[1]]
        ), bindings
        banks[key] = fd, size, bindings
        return banks[key]

    continuation, first_output = {}, {}
    for case in ("ffn", "attention", "layer", "poison", "decode"):
        n, position = (1, 65) if case == "decode" else (2, 63)
        active_inputs = decode_inputs if case == "decode" else inputs
        payload = array.array("f", (x for row in active_inputs for x in row)).tobytes()
        poison = struct.pack("<I", 0x7FC11223 if case == "poison" else 0x7FC55667)
        residual = payload + poison * ((t - n) * e + 64)
        epoch = bytearray([0x5A]) * 2096
        struct.pack_into("<3i", epoch, 0, 1, 1, 1)
        struct.pack_into("<5i", epoch, 12, n, position, 0, 0, 0)
        epoch = bytes(epoch)
        root = "validated.qwen3." + (
            "layer0" if case in ("layer", "poison", "decode") else case + "0"
        )
        expected_rows, expected_state = expected_by_case[case]
        expected_flat = [x for row in expected_rows for x in row]
        for kind, path in modules.items():
            before = continuation[kind] if case == "decode" else initial_cache
            work = args.output / (case + "-" + kind)
            work.mkdir(exist_ok=True)
            fd, length, bindings = bank(kind, root)
            for name, data in [
                ("residual", residual),
                ("epoch", epoch),
                ("map", map_data),
                ("cache", before),
            ]:
                save(work / (name + ".npy"), data)
            test = work / "check.loom"
            test.write_text(
                f"""command.decl @{root}() launch(%p:buffer,%r:buffer,%c:buffer,%m:buffer,%k:buffer)
func.def @noop(%p:buffer,%r:buffer,%c:buffer,%m:buffer,%k:buffer) {{ func.return }}
check.scenario @independent_layer {{
 check.trial[1](%trial:index,%entropy:check.entropy) {{
  %p = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>
  %r = check.file.read.npy path("{work/'residual.npy'}") : tensor<{len(residual)}xi8>
  %c = check.file.read.npy path("{work/'epoch.npy'}") : tensor<{len(epoch)}xi8>
  %m = check.file.read.npy path("{work/'map.npy'}") : tensor<{len(map_data)}xi8>
  %k = check.file.read.npy path("{work/'cache.npy'}") : tensor<{len(before)}xi8>
  check.compare<@{root},@noop>(%p,%r,%c,%m,%k) : (tensor<{length}xi8>,tensor<{len(residual)}xi8>,tensor<{len(epoch)}xi8>,tensor<{len(map_data)}xi8>,tensor<{len(before)}xi8>) -> () {{
   check.expect.bitwise actual(%p) expected(%p) : tensor<{length}xi8>
   check.expect.bitwise actual(%c) expected(%c) : tensor<{len(epoch)}xi8>
   check.expect.bitwise actual(%m) expected(%m) : tensor<{len(map_data)}xi8>
  }}
 }}
 check.return
}}
"""
            )
            capture = work / "capture"
            capture.mkdir(exist_ok=True)
            command = [
                str(args.runner),
                str(test),
                "--library=" + str(path),
                "--case=@independent_layer",
                "--device=amdgpu",
            ]
            start = time.monotonic()
            with (
                args.gpu_lock.open("a") as lock,
                (work / "execution.log").open("w") as log,
            ):
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(
                    command,
                    env=dict(env, LOOM_FFN_CAPTURE_DIR=str(capture)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(fd,),
                )
            record["runs"].append(
                {
                    "case": case,
                    "format": kind,
                    "command": command,
                    "exit": result.returncode,
                    "seconds": time.monotonic() - start,
                    "parameter_bindings": bindings,
                }
            )
            publish()
            assert result.returncode == 0, (case, kind, work / "execution.log")
            actual = load(capture / "product0_call0_after_arg1.npy")
            actual_cache = load(capture / "product0_call0_after_arg4.npy")
            assert actual[n * e * 4 :] == residual[n * e * 4 :], (
                case,
                kind,
                "inactive residual/guard",
            )
            comparison, passed = close(
                floats(actual[: n * e * 4]),
                expected_flat,
                case + "-" + kind + "-residual",
            )
            record["comparisons"].append(comparison)
            publish()
            assert passed, comparison
            intervals = []
            if case != "ffn":
                for plane in (0, 1):
                    for pos in range(position, position + n):
                        begin = cache_offset(plane, pos)
                        intervals.append((begin, begin + kv * 2))
                a, b = [], []
                for begin, end in intervals:
                    a += list(struct.unpack("<" + "e" * kv, actual_cache[begin:end]))
                    b += list(struct.unpack("<" + "e" * kv, expected_state[begin:end]))
                comparison, passed = close(a, b, case + "-" + kind + "-written-cache")
                record["comparisons"].append(comparison)
                publish()
                assert passed, comparison
            cursor = 0
            for begin, end in sorted(intervals):
                assert actual_cache[cursor:begin] == before[cursor:begin], (
                    case,
                    kind,
                    "cache guard",
                    cursor,
                    begin,
                )
                cursor = end
            assert actual_cache[cursor:] == before[cursor:], (
                case,
                kind,
                "cache suffix",
            )
            if kind == "source":
                source_outputs[case] = (actual, actual_cache)
            else:
                assert source_outputs[case] == (actual, actual_cache), (
                    case,
                    "SOURCE/BC mismatch",
                )
            if case == "layer":
                continuation[kind] = actual_cache
                first_output[kind] = actual[: n * e * 4], actual_cache
            if case == "poison":
                assert first_output[kind] == (
                    actual[: n * e * 4],
                    actual_cache,
                ), "poisoned inactive-row influence"
            comparison = {
                "name": case + "-" + kind + "-ownership",
                "passed": True,
                "inactive_residual_guards_exact": True,
                "unwritten_cache_planes_guards_exact": True,
                "output_sha256": digest(actual),
                "cache_sha256": digest(actual_cache),
            }
            record["comparisons"].append(comparison)
            publish()
            save(work / "actual-active-residual.npy", actual[: n * e * 4])
            save(
                work / "actual-written-cache.npy",
                b"".join(actual_cache[begin:end] for begin, end in intervals),
            )
            for captured in capture.glob("*.npy"):
                captured.unlink()
            for name in ("residual", "epoch", "map", "cache"):
                (work / (name + ".npy")).unlink()
            print(case, kind, "PASS", flush=True)
    for fd, _, _ in banks.values():
        os.close(fd)
    assert record["modules"] == {
        name: digest(path.read_bytes()) for name, path in modules.items()
    }
    assert record["helpers"] == {
        str(p.relative_to(source)): digest(p.read_bytes())
        for p in (helper / "fixtures.py", helper / "goldens.py")
    }
    record["status"] = "passed"
    record["limits"] = (
        "First layer actual fixture/stated arithmetic only; independent fullmodel fidelity and performance not established."
    )
    publish()


if __name__ == "__main__":
    main()
