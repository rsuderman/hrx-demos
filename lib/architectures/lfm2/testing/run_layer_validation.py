# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Bounded LFM2 layer/state arithmetic using reflected sparse Q4/Q6 weights.

Uses the existing parameter reflector and numerical testbench runner. Inputs and
small scalar expectations are generated at runtime; no checkpoint activations,
reference model, template or Serve runtime is used. Central owns execution.
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

E, ROWS, EPS = 2048, 4, 9.999999747378752e-6
DOWN6 = {0, 1, 4, 7, 10, 13, 14, 15}
CONV_ORD = {layer: i for i, layer in enumerate((0, 1, 3, 4, 6, 7, 9, 11, 13, 15))}
ATTN_ORD = {layer: i for i, layer in enumerate((2, 5, 8, 10, 12, 14))}
f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
f16 = lambda x: struct.unpack("<e", struct.pack("<e", x))[0]


def header(size):
    text = repr({"descr": "|i1", "fortran_order": False, "shape": (size,)})
    text += " " * ((64 - (10 + len(text) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(text)) + text.encode()


def save(path, data):
    path.write_bytes(header(len(data)) + data)


def load(path):
    data = path.read_bytes()
    width = 2 if data[6] == 1 else 4
    size = int.from_bytes(data[8 : 8 + width], "little")
    ast.literal_eval(data[8 + width : 8 + width + size].decode())
    return data[8 + width + size :]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def block(tag, coefficient):
    # Logical first256 coefficients all coefficient/256; all other blocks zero.
    # Independent GGUF encoders, not copied GPU dequantization arithmetic.
    d = struct.pack("<e", coefficient / 256)
    if tag == 12:
        return d + bytes(2) + bytes([1] * 4 + [0] * 4 + [1] * 4) + bytes([0x11]) * 128
    assert tag == 14
    return bytes([0x11]) * 128 + bytes([0xAA]) * 64 + bytes([1]) * 16 + d


def weight_recipe(fd, origin, name, size):
    if name == "token_embd_norm.weight":
        data = struct.pack("<2048f", *([0.5] * 1024 + [2.0] * 1024))
        assert len(data) == size
        os.pwrite(fd, data, origin)
        return
    if name.endswith("_norm.weight"):
        gamma = (
            2.0
            if name.endswith(".attn_norm.weight")
            else 0.5 if name.endswith(".ffn_norm.weight") else 1.0
        )
        assert size % 4 == 0
        os.pwrite(
            fd,
            struct.pack("<" + str(size // 4) + "f", *([gamma] * (size // 4))),
            origin,
        )
        return
    if name.endswith(".shortconv.conv.weight"):
        assert size == E * 3 * 4
        os.pwrite(fd, struct.pack("<3f", 0.25, 0.5, 1.0) * E, origin)
        return
    if name == "token_embd.weight":
        stride = E // 256 * 210
        assert size == 65536 * stride
        # Final gamma makes row9 > row7. Missing gamma would tie at row7.
        os.pwrite(fd, block(14, 1), origin + 7 * stride)
        os.pwrite(fd, block(14, 1), origin + 9 * stride + 4 * 210)
        return
    match = re.fullmatch(r"blk\.(\d+)\.(.+)\.weight", name)
    assert match, name
    layer, role = int(match[1]), match[2]
    tag = (
        14
        if (role == "ffn_down" and layer in DOWN6)
        or (role == "attn_v" and layer in (8, 14))
        else 12
    )
    width = 8192 if role == "ffn_down" else E
    stride = width // 256 * {12: 144, 14: 210}[tag]
    assert size % stride == 0, (name, size, stride)
    for row in range(size // stride):
        coefficient = (1, 2, 0.5)[row // E] if role == "shortconv.in_proj" else 1
        os.pwrite(fd, block(tag, coefficient), origin + row * stride)


def projected(x):
    # Uniform positive activation in the first256 columns: canonical Q8_1
    # scale rounds to F16; code127, zero dmin and weight coefficient1/256.
    return f32(127 * f16(f32(x / 127)))


def normalized_first(pair, gamma):
    a, b = pair
    mean = f32(f32(a * a) * 0.5 + f32(b * b) * 0.5)
    return f32(f32(a / math.sqrt(mean + EPS)) * gamma)


def rope(value, position):
    norm = f32(value / math.sqrt(f32(value * value) + EPS))
    angles = [position * 1000000 ** (-i / 32) for i in range(32)]
    return [f32(norm * (math.cos(x) - math.sin(x))) for x in angles] + [
        f32(norm * (math.cos(x) + math.sin(x))) for x in angles
    ]


def oracle(values, layers, histories, attention_history, base):
    if not values:
        return [], dict(histories), dict(attention_history)
    result = list(values)
    expected_history = dict(histories)
    expected_attention = {k: dict(v) for k, v in attention_history.items()}
    for layer in layers:
        normal = [normalized_first(x, 2.0) for x in result]
        if layer in CONV_ORD:
            h = [0.0, 0.0] if base == 0 else list(expected_history[layer])
            mixed = []
            for x in normal:
                b, c, raw = projected(x), f32(2 * projected(x)), f32(0.5 * projected(x))
                current = f32(b * raw)
                conv = f32(f32(f32(0.25 * h[0]) + f32(0.5 * h[1])) + current)
                mixed.append(projected(f32(c * conv)))
                h = [h[1], current]
            expected_history[layer] = h
        else:
            stored = {} if base == 0 else expected_attention[layer]
            for row, x in enumerate(normal):
                q = rope(projected(x), base + row)
                stored[base + row] = ([f16(y) for y in q], f16(projected(x)))
            mixed = []
            for row, x in enumerate(normal):
                position = base + row
                q = rope(projected(x), position)
                scores = [
                    sum(a * b for a, b in zip(q, stored[j][0])) / 8
                    for j in range(position + 1)
                ]
                maximum = max(scores)
                probabilities = [math.exp(score - maximum) for score in scores]
                value = sum(
                    p * stored[j][1] for j, p in enumerate(probabilities)
                ) / sum(probabilities)
                mixed.append(projected(f32(value)))
            expected_attention[layer] = stored
        result = [(f32(a + y), f32(b + y)) for (a, b), y in zip(result, mixed)]
        out = []
        for pair in result:
            g = projected(normalized_first(pair, 0.5))
            activation = f32(f32(g / (1 + math.exp(-g))) * g)
            down = projected(activation)
            out.append(tuple(f32(x + down) for x in pair))
        result = out
    return result, expected_history, expected_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("emitted", "gguf", "reflector", "runner", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--actual-weights",
        action="store_true",
        help="Read actual reflected GGUF ranges; state/ABI execution, no fidelity oracle",
    )
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((args.emitted / "layout.json").read_text())
    assert geometry == {
        "rows": 4,
        "layers": 16,
        "pool": 256,
        "hidden": 2048,
        "kv": 512,
        "cache_base": 1284,
        "history_base": 3147012,
        "map_origin": 2048,
        "cache_bytes": 3310852,
        "map_bytes": 2064,
    }
    env = dict(os.environ, PARAMETER_GGUF=str(args.gguf), PYTHONNOUSERSITE="1")
    record = {
        "geometry": geometry,
        "recipe": (
            "actual GGUF state/ABI"
            if args.actual_weights
            else "sparse independent Q4/Q6 first-block rows"
        ),
        "runs": [],
    }
    modules = [
        args.emitted / "configured_layer.loom",
        args.emitted / "configured_layer.loombc",
    ]
    parity = {}
    for module in modules:
        for suffix, layers in [
            ("layer0", [0]),
            ("layer2", [2]),
            ("layer8", [8]),
            ("layer14", [14]),
            ("conv_attention_conv", [1, 2, 3]),
            ("head", []),
        ]:
            root = "validated.lfm2." + suffix
            prefix = args.output / (module.suffix[1:] + "-" + suffix)
            command = [str(args.reflector), root, str(module)]
            reflected = subprocess.run(
                command, env=env, capture_output=True, text=True, check=True
            )
            prefix.with_suffix(".reflection.log").write_text(reflected.stderr)
            prefix.with_suffix(".parameters.tsv").write_text(reflected.stdout)
            roots = re.findall(
                r"ROOT (\d+) length=(\d+) alignment=(\d+)", reflected.stderr
            )
            assert len(roots) == 1 and roots[0][0] == "0", roots
            length = int(roots[0][1])
            fd = os.memfd_create("lfm2-sparse-weights", os.MFD_CLOEXEC)
            preamble = header(length)
            os.ftruncate(fd, len(preamble) + length)
            os.pwrite(fd, preamble, 0)
            payloads = []
            with args.gguf.open("rb", buffering=0) as checkpoint:
                for line in reflected.stdout.splitlines():
                    name, offset, size, bank, source = line.split("\t")
                    assert bank == "0"
                    if args.actual_weights:
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
                        payloads.append(
                            {
                                "name": name,
                                "source": int(source),
                                "size": int(size),
                                "sha256": digest.hexdigest(),
                            }
                        )
                    else:
                        weight_recipe(fd, len(preamble) + int(offset), name, int(size))
            poisoned = bytearray([0x33]) * (geometry["cache_bytes"] + 64)
            poisoned[geometry["history_base"] : geometry["cache_bytes"]] = struct.pack(
                "<I", 0x7FC00000
            ) * (163840 // 4)
            initial_cache = bytes(poisoned)
            before_cache = initial_cache
            histories, attention_history = {}, {}
            fresh_state = None
            cases = (
                [("head", 3, 0)]
                if suffix == "head"
                else [
                    ("fresh", 3, 0),
                    ("append", 1, 3),
                    ("reset", 3, 0),
                    ("inactive", 0, 4),
                ]
            )
            if suffix in ("layer0", "layer2", "conv_attention_conv"):
                cases += [("out-of-context", 1, 256), ("cross-context", 3, 254)]
            for label, active, base in cases:
                work = args.output / (prefix.name + "-" + label)
                work.mkdir(exist_ok=True)
                scalars = [1 + base * 0.25 + r * 0.25 for r in range(ROWS)]
                # Actual layers RMS-normalize away a radial scalar change.
                # Change direction across channel halves on each logical token
                # so append exercises a new B*X sample rather than an identical
                # quantized normalized row. Sparse-oracle recipe stays fixed.
                pairs = [
                    (
                        (x, x * (0.5 + 0.125 * (base + r)))
                        if args.actual_weights
                        else (x, 0.5 * x)
                    )
                    for r, x in enumerate(scalars)
                ]
                residual = (
                    b"".join(
                        struct.pack("<f", a) * 1024 + struct.pack("<f", b) * 1024
                        for a, b in pairs
                    )
                    + bytes([0x55]) * 64
                )
                if label == "reset":
                    before_cache = initial_cache
                    histories, attention_history = {}, {}
                control = bytearray([0x71]) * 64
                struct.pack_into("<5i", control, 12, active, base, 0, 0, 0)
                if suffix == "head":
                    control = bytearray(struct.pack("<3i", active, 0, 0))
                page_map = bytearray([0x59]) * (geometry["map_bytes"] + 64)
                for page, physical in enumerate((2, 0, 3, 1)):
                    struct.pack_into("<i", page_map, 2048 + page * 4, physical)
                winner = bytes([0x66]) * 16
                arguments = [("r", residual), ("c", bytes(control))]
                arguments += (
                    [("o", winner)]
                    if suffix == "head"
                    else [("m", bytes(page_map)), ("k", before_cache)]
                )
                for name, data in arguments:
                    save(work / (name + ".npy"), data)
                decl = ",".join(
                    "%" + n + ":buffer" for n in ["p"] + [n for n, _ in arguments]
                )
                names = ["p"] + [n for n, _ in arguments]
                sizes = [length] + [len(data) for _, data in arguments]
                fixture = f"command.decl @{root}() launch({decl})\nfunc.def @noop({decl}) {{ func.return }}\ncheck.scenario @lfm2_layer {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n"
                for name, size in zip(names, sizes):
                    path = (
                        f"/proc/self/fd/{fd}"
                        if name == "p"
                        else str(work / (name + ".npy"))
                    )
                    fixture += f'  %{name} = check.file.read.npy path("{path}") : tensor<{size}xi8>\n'
                operands = ",".join("%" + n for n in names)
                types = ",".join(f"tensor<{n}xi8>" for n in sizes)
                fixture += (
                    f"  check.compare<@{root},@noop>({operands}) : ({types}) -> () {{\n"
                )
                for name in (["p", "r", "c"] if suffix == "head" else ["p", "c", "m"]):
                    size = sizes[names.index(name)]
                    fixture += f"   check.expect.bitwise actual(%{name}) expected(%{name}) : tensor<{size}xi8>\n"
                fixture += "  }\n }\n check.return\n}\n"
                (work / "fixture.loom").write_text(fixture)
                capture = work / "capture"
                capture.mkdir(exist_ok=True)
                command = [
                    str(args.runner),
                    str(work / "fixture.loom"),
                    "--library=" + str(module),
                    "--case=@lfm2_layer",
                    "--device=amdgpu",
                ]
                with (
                    args.gpu_lock.open("a") as lock,
                    (work / "execution.log").open("w") as log,
                ):
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    run = subprocess.run(
                        command,
                        env=dict(env, LOOM_FFN_CAPTURE_DIR=str(capture)),
                        pass_fds=(fd,),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                assert run.returncode == 0, (root, module, label, work)
                if suffix == "head":
                    output = load(capture / "product0_call0_after_arg3.npy")
                    winner_id = struct.unpack_from("<i", output)[0]
                    assert (
                        (0 <= winner_id < 65536)
                        if args.actual_weights
                        else winner_id == 9
                    ), output.hex()
                else:
                    output = load(capture / "product0_call0_after_arg1.npy")
                    cache = load(capture / "product0_call0_after_arg4.npy")
                    assert output[active * E * 4 :] == residual[active * E * 4 :]
                    effective = active if 0 <= base and base + active <= 256 else 0
                    assert output[effective * E * 4 :] == residual[effective * E * 4 :]
                    observed = array.array("f")
                    observed.frombytes(output[: effective * E * 4])
                    assert all(math.isfinite(x) for x in observed)
                    if effective:
                        assert (
                            output[: effective * E * 4] != residual[: effective * E * 4]
                        )
                    if not args.actual_weights:
                        expected, histories, attention_history = oracle(
                            pairs[:effective],
                            layers,
                            histories,
                            attention_history,
                            base,
                        )
                        for row, pair in enumerate(expected):
                            for half, value in enumerate(pair):
                                segment = observed[
                                    row * E + half * 1024 : row * E + (half + 1) * 1024
                                ]
                                assert all(
                                    abs(x - value) <= 0.002 * max(1, abs(value))
                                    for x in segment
                                ), (root, label, row, half, value, segment[0])
                    allowed = []
                    if effective:
                        for layer in layers:
                            if layer in CONV_ORD:
                                origin = (
                                    geometry["history_base"] + CONV_ORD[layer] * 16384
                                )
                                allowed.append((origin, origin + 16384))
                                actual_history = struct.unpack_from(
                                    "<4096f", cache, origin
                                )
                                assert all(math.isfinite(x) for x in actual_history)
                                if args.actual_weights:
                                    if label == "append":
                                        assert (
                                            cache[origin : origin + 8192]
                                            == before_cache[
                                                origin + 8192 : origin + 16384
                                            ]
                                        ), (root, "append must carry prior newest B*X")
                                    assert (
                                        cache[origin : origin + 16384]
                                        != before_cache[origin : origin + 16384]
                                    )
                                else:
                                    for r, value in enumerate(histories[layer]):
                                        assert all(
                                            abs(x - value) < 0.001
                                            for x in actual_history[r * E : (r + 1) * E]
                                        )
                            else:
                                for plane in (
                                    2 * ATTN_ORD[layer],
                                    2 * ATTN_ORD[layer] + 1,
                                ):
                                    for position in range(base, base + effective):
                                        physical = (2, 0, 3, 1)[
                                            position // 64
                                        ] * 64 + position % 64
                                        origin = (
                                            geometry["cache_base"]
                                            + plane * 256 * 512 * 2
                                            + physical * 512 * 2
                                        )
                                        allowed.append((origin, origin + 1024))
                    cursor = 0
                    for begin, end in sorted(allowed):
                        assert cache[cursor:begin] == before_cache[cursor:begin], (
                            root,
                            label,
                            cursor,
                            begin,
                        )
                        cursor = end
                    assert cache[cursor:] == before_cache[cursor:]
                    before_cache = cache
                if label == "fresh":
                    fresh_state = (sha(output), sha(before_cache))
                if label == "reset":
                    assert fresh_state == (sha(output), sha(before_cache)), (
                        "reset replay mismatch",
                        root,
                    )
                parity_key = (suffix, label)
                state = (sha(output), sha(before_cache))
                if module.suffix == ".loom":
                    parity[parity_key] = state
                else:
                    assert state == parity[parity_key], (
                        "SOURCEBC mismatch",
                        parity_key,
                    )
                record["runs"].append(
                    {
                        "root": root,
                        "module": str(module),
                        "module_sha256": sha(module.read_bytes()),
                        "case": label,
                        "command": command,
                        "exit": run.returncode,
                        "result_sha256": sha(output),
                        "independent_scalar_oracle": not args.actual_weights
                        and suffix != "head",
                        "actual_weight_payloads": payloads,
                        "gamma_sensitive_tied_winner": not args.actual_weights
                        and suffix == "head",
                    }
                )
                (args.output / "results.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                # Logs/recipes/hashes are evidence; NPY payloads are disposable.
                for path in capture.glob("*.npy"):
                    path.unlink()
                for name, _ in arguments:
                    (work / (name + ".npy")).unlink()
            os.close(fd)
    record["passed"] = True
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
