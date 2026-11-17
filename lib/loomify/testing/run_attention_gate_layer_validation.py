# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Finite actual Spark2.5/LFM3 layer state gate; independent math is component-owned.

Reuses maintained raw-GGUF bank/NPY capture conventions, no activation corpus.
Run centrally in its isolated venv with the shared GPU lock.
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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def npy_header(length):
    text = repr({"descr": "|i1", "fortran_order": False, "shape": (length,)})
    text += " " * ((64 - (10 + len(text) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(text)) + text.encode()


def save(path, data):
    path.write_bytes(npy_header(len(data)) + data)


def load(path):
    data = path.read_bytes()
    assert data[:6] == b"\x93NUMPY"
    width = 2 if data[6] == 1 else 4
    length = int.from_bytes(data[8 : 8 + width], "little")
    info = ast.literal_eval(data[8 + width : 8 + width + length].decode())
    assert info["descr"] in ("|i1", "<i1", "|u1", "<u1") and not info["fortran_order"]
    return data[8 + width + length :]


def floats(data):
    result = array.array("f")
    result.frombytes(data)
    assert all(math.isfinite(x) for x in result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("emitted", "gguf", "reflector", "runner", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    geometry = json.loads((args.emitted / "layout.json").read_text())
    family, hidden, rows = geometry["family"], geometry["hidden"], geometry["rows"]
    assert family in ("lfm3", "spark2_5") and rows == 3
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PARAMETER_GGUF=str(args.gguf))
    record = {"passed": False, "geometry": geometry, "runs": []}
    variants = {}
    for form in ("loom", "loombc"):
        module = args.emitted / ("configured_layers." + form)
        for block in geometry["blocks"]:
            layer, ordinal, conv = (
                block["layer"],
                block["ordinal"],
                block["convolution"],
            )
            root = f"validated.{family}.layer{layer}"
            work = args.output / f"{form}-layer{layer}"
            work.mkdir(exist_ok=True)
            reflected = work / "parameters.tsv"
            with reflected.open("w") as out, (work / "reflection.log").open("w") as err:
                subprocess.run(
                    [str(args.reflector), root, str(module)],
                    env=env,
                    stdout=out,
                    stderr=err,
                    check=True,
                )
            roots = re.findall(
                r"ROOT (\d+) length=(\d+) alignment=(\d+)",
                (work / "reflection.log").read_text(),
            )
            assert len(roots) == 1 and roots[0][0] == "0"
            length = int(roots[0][1])
            fd = os.memfd_create("attention-gate-layer-weights", os.MFD_CLOEXEC)
            try:
                header = npy_header(length)
                os.ftruncate(fd, len(header) + length)
                os.pwrite(fd, header, 0)
                sizes = {}
                with args.gguf.open("rb", buffering=0) as source:
                    for line in reflected.read_text().splitlines():
                        name, offset, size, bank, source_origin = line.split("\t")
                        assert int(bank) == 0
                        sizes[name] = int(size)
                        source.seek(int(source_origin))
                        remaining, dest = int(size), len(header) + int(offset)
                        while remaining:
                            data = source.read(min(remaining, 1024 * 1024))
                            assert data and os.pwrite(fd, data, dest) == len(data)
                            remaining -= len(data)
                            dest += len(data)
                assert sorted(sizes) == sorted(block["bindings"]), (
                    root,
                    "exact weight roles",
                )
                gate_name = f"blk.{layer}.attn_gate.weight"
                if conv:
                    assert gate_name not in sizes
                else:
                    assert sizes[gate_name] == hidden // 256 * 144 * 16
                state = bytes([91]) * (geometry["state_bytes"] + 256)
                first = None
                for step, position in (("fresh", 0), ("append", 1), ("reset", 0)):
                    case = work / step
                    case.mkdir(exist_ok=True)
                    if step == "reset":
                        assert first is not None
                    input_row = b"".join(
                        struct.pack("<f", ((c % 31) - 15) / 64 + position / 128)
                        for c in range(hidden)
                    )
                    residual = input_row + struct.pack("<f", -31.0) * (
                        (rows - 1) * hidden + 64
                    )
                    epoch = bytearray(4192)
                    struct.pack_into("<8i", epoch, 0, 1, 1, 0, 1, position, 0, 0, -1)
                    page_map = bytearray(2052)
                    struct.pack_into("<i", page_map, 2048, 0)
                    buffers = [residual, bytes(epoch), bytes(page_map), state]
                    for i, data in enumerate(buffers, 1):
                        save(case / f"arg{i}.npy", data)
                    declarations = ",".join(f"%a{i}:buffer" for i in range(5))
                    tensors = ",".join(
                        [f"tensor<{length}xi8>"]
                        + [f"tensor<{len(x)}xi8>" for x in buffers]
                    )
                    fixture = f"command.decl @{root}() launch({declarations})\nfunc.def @noop({declarations}) {{ func.return }}\ncheck.scenario @raw_layer {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n"
                    fixture += f'  %a0 = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>\n'
                    for i, data in enumerate(buffers, 1):
                        fixture += f'  %a{i} = check.file.read.npy path("{case / f"arg{i}.npy"}") : tensor<{len(data)}xi8>\n'
                    fixture += f"  check.compare<@{root},@noop>(%a0,%a1,%a2,%a3,%a4) : ({tensors}) -> () {{\n"
                    for i, size in ((0, length), (2, len(epoch)), (3, len(page_map))):
                        fixture += f"   check.expect.bitwise actual(%a{i}) expected(%a{i}) : tensor<{size}xi8>\n"
                    fixture += "  }\n }\n check.return\n}\n"
                    (case / "check.loom").write_text(fixture)
                    capture = case / "capture"
                    capture.mkdir(exist_ok=True)
                    command = [
                        str(args.runner),
                        str(case / "check.loom"),
                        "--library=" + str(module),
                        "--case=@raw_layer",
                        "--device=amdgpu",
                    ]
                    with (
                        args.gpu_lock.open("a") as lock,
                        (case / "execution.log").open("w") as log,
                    ):
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        result = subprocess.run(
                            command,
                            env=dict(env, LOOM_FFN_CAPTURE_DIR=str(capture)),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            pass_fds=(fd,),
                        )
                    assert result.returncode == 0, case / "execution.log"
                    output = load(capture / "product0_call0_after_arg1.npy")
                    next_state = load(capture / "product0_call0_after_arg4.npy")
                    active = output[: hidden * 4]
                    floats(active)
                    assert active != input_row
                    assert output[hidden * 4 :] == residual[hidden * 4 :], (
                        root,
                        step,
                        "inactive residual",
                    )
                    if conv:
                        begin = geometry["history_base"] + ordinal * 6 * hidden * 4
                        allowed = [(begin, begin + 6 * hidden * 4)]
                        floats(next_state[begin : allowed[0][1]])
                    else:
                        plane_bytes = geometry["pool"] * geometry["kv"] * 2
                        begin = geometry["cache_base"] + ordinal * 2 * plane_bytes
                        token_bytes = geometry["kv"] * 2
                        allowed = [
                            (
                                begin + p * plane_bytes + position * token_bytes,
                                begin + p * plane_bytes + (position + 1) * token_bytes,
                            )
                            for p in (0, 1)
                        ]
                    cursor = 0
                    for begin, end in allowed:
                        assert next_state[cursor:begin] == state[cursor:begin], (
                            root,
                            step,
                            "nonselected state",
                        )
                        if step != "reset":
                            assert next_state[begin:end] != state[begin:end]
                        cursor = end
                    assert next_state[cursor:] == state[cursor:]
                    selected = b"".join(next_state[a:b] for a, b in allowed)
                    if step == "fresh":
                        first = (active, selected)
                    if step == "reset":
                        assert first == (active, selected), (root, "dirty reset replay")
                    identity = (digest(output), digest(next_state))
                    key = (layer, step)
                    if form == "loom":
                        variants[key] = identity
                    else:
                        assert variants[key] == identity, (
                            root,
                            step,
                            "SOURCE/BC parity",
                        )
                    record["runs"].append(
                        {
                            "layer": layer,
                            "ordinal": ordinal,
                            "window": block["window"],
                            "convolution": conv,
                            "form": form,
                            "step": step,
                            "active_rows": 1,
                            "static_rows": rows,
                            "gate_bytes": sizes.get(gate_name),
                            "module_sha256": digest(module.read_bytes()),
                            "output_sha256": identity[0],
                            "state_sha256": identity[1],
                        }
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
                    state = next_state
                    for path in capture.glob("*.npy"):
                        path.unlink()
                    for path in case.glob("arg*.npy"):
                        path.unlink()
            finally:
                os.close(fd)
    record["passed"] = True
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
