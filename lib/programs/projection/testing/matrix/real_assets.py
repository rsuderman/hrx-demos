#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Execute actual GGUF projection slices against independent staged-half goldens.

The caller supplies the existing tensor-header index, native tools and isolated
runtime environment. Weights are read-only; all generated inputs remain external.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import input_value
from goldens import f32, scaled_matrix, gated_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npy(path, data, descriptor, count):
    header = repr({"descr": descriptor, "fortran_order": False, "shape": (count,)})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header = (header + " " * padding + "\n").encode()
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data
    )


def floats(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x93NUMPY\x01\x00"
        header = ast.literal_eval(
            stream.read(struct.unpack("<H", stream.read(2))[0]).decode()
        )
        assert header["descr"] == "<f4" and not header["fortran_order"]
        data = stream.read()
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def shape(value):
    try:
        capacity, active = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be CAPACITY:ACTIVE") from error
    if not 17 <= capacity <= 512 or not 0 <= active <= capacity:
        raise argparse.ArgumentTypeError(
            "require 17 <= capacity <= 512 and 0 <= active <= capacity"
        )
    return capacity, active


def main():
    parser = argparse.ArgumentParser()
    for flag in [
        "gguf",
        "tensor-index",
        "source-root",
        "output",
        "runner",
        "loom-link",
        "gpu-lock",
    ]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument(
        "--shape",
        type=shape,
        action="append",
        help="scaled matrix CAPACITY:ACTIVE, repeatable; default 17:2",
    )
    parser.add_argument(
        "--wide-q6-down",
        action="store_true",
        help="exact audited Qwen32 full-shape C1/C17 down witness",
    )
    parser.add_argument("--provider-text", type=Path)
    parser.add_argument("--provider-bc", type=Path)
    parser.add_argument(
        "--matrix-proof-root",
        type=Path,
        help="strict-normalized full-shape matrix proof directory for wide-Q6 witness",
    )
    args = parser.parse_args()
    if not os.environ.get("LOOM_SERVE_DIR"):
        parser.error(
            "set the isolated validation instance LOOM_SERVE_DIR before invocation"
        )
    if args.wide_q6_down:
        from wide_q6_real_assets import run

        return run(args)
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.tensor_index.read_text())
    tensors = {entry["name"]: entry for entry in index["tensors"]}
    records = []

    def weight(name):
        tensor = tensors[name]
        k, n = tensor["dimensions"]
        tag = tensor["storage_type"]
        assert tag in (12, 14) and n >= 64 and k % 256 == 0
        size = 64 * (k // 256) * {12: 144, 14: 210}[tag]
        with args.gguf.open("rb") as stream:
            stream.seek(tensor["absolute_offset"])
            data = stream.read(size)
        assert len(data) == size
        return (
            k,
            tag,
            data,
            {
                "tensor": name,
                "full_dimensions": [k, n],
                "slice_dimensions": [k, 64],
                "tag": tag,
                "offset": tensor["absolute_offset"],
                "bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    sources = [
        args.source_root / entry
        for entry in [
            "lib/platforms/amdgpu/gfx11/targets.loom",
            "lib/platforms/amdgpu/gfx11/projection/q4k.loom",
            "lib/platforms/amdgpu/gfx11/projection/q6k.loom",
            "lib/platforms/amdgpu/gfx11/projection/gated_q4k.loom",
            "lib/kernels/projection/f16_staging.loom",
            "lib/kernels/projection/qk_common.loom",
            "lib/kernels/projection/scaled_f16.loom",
        ]
    ]
    source_hashes = {str(p): sha(p) for p in sources}
    libraries = []
    for fmt, extension in [("text", "loom"), ("bc", "loombc")]:
        target = args.output / ("selected-matrix." + extension)
        cmd = [
            str(args.loom_link),
            "--mode=merge",
            "--allow-unresolved",
            "--to=" + fmt,
            *map(str, sources),
            "--output=" + str(target),
        ]
        with (args.output / ("merge-" + fmt + ".log")).open("w") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=60
            )
        assert result.returncode == 0, cmd
        libraries.append(target)
    cases = [
        ("q4", "blk.0.attn_q.weight", False),
        ("q6", "blk.0.attn_v.weight", False),
        ("q6_down", "blk.0.ffn_down.weight", False),
        ("gated", "blk.0.ffn_gate.weight", True),
    ]
    with args.gpu_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, tensor, gated, t, active in [
            (*case, capacity, active)
            for case in cases
            for capacity, active in (args.shape or [(17, 2)])
        ]:
            directory = args.output / f"{name}_t{t}_a{active}"
            directory.mkdir()
            k, tag, raw, binding = weight(tensor)
            up = raw
            bindings = [binding]
            if gated:
                k1, tag1, up, other = weight("blk.0.ffn_up.weight")
                assert (k1, tag1) == (k, 12)
                bindings.append(other)
            tp = (t + 63) // 64 * 64
            n = 64
            inputs = [
                [input_value(row, column, False) for column in range(k)]
                for row in range(active)
            ]
            expected = (
                (
                    gated_matrix(inputs, raw, up, n)
                    if gated
                    else scaled_matrix(inputs, raw, tag, n)
                )
                if active
                else []
            )
            flat = [x for row in inputs for x in row] + [float("nan")] * (
                (t - active) * k
            )
            golden = [x for row in expected for x in row] + [0.0] * ((tp - active) * n)
            wb = len(raw)
            write_npy(
                directory / "input.npy",
                struct.pack("<" + "f" * len(flat), *flat),
                "<f4",
                len(flat),
            )
            write_npy(directory / "weight.npy", raw, "|i1", wb)
            write_npy(directory / "up.npy", up, "|i1", wb)
            write_npy(
                directory / "count.npy", struct.pack("<3i", active, 0, 0), "<i4", 3
            )
            write_npy(
                directory / "expected.npy",
                struct.pack("<" + "f" * len(golden), *golden),
                "<f4",
                len(golden),
            )
            atol, rtol = (0.0005, 0.002) if gated else (0.0001, 0.0005)
            kernel = (
                "q4k_scaled_f16_swiglu"
                if gated
                else ("q4k" if tag == 12 else "q6k") + "_scaled_f16_store"
            )
            declaration = (
                "kernel.decl @llm.platform."
                + kernel
                + "(%rows:index,%K:index,%N:index) launch(%rows:index,%K:index,%N:index,%count:buffer,%input:buffer,%scales:buffer,%weight:buffer,"
                + ("%up:buffer," if gated else "")
                + "%output:buffer)\n"
            )
            text = (
                declaration
                + "kernel.decl @llm.projection.prepare_scaled_f16(%T:index,%Tp:index,%K:index) launch(%T:index,%Tp:index,%K:index,%origin:offset,%count:buffer,%input:buffer,%scales:buffer,%output:buffer)\n"
            )
            text += f"""check.case public @real_matrix {{
  %T = check.literal value({t}) : index
  %Tp = check.literal value({tp}) : index
  %K = check.literal value({k}) : index
  %N = check.literal value({n}) : index
  %origin = check.literal value(0) : offset
  %input = check.file.read.npy path("{directory/'input.npy'}") : tensor<{t*k}xf32>
  %input_copy = check.file.read.npy path("{directory/'input.npy'}") : tensor<{t*k}xf32>
  %weight = check.file.read.npy path("{directory/'weight.npy'}") : tensor<{wb}xi8>
  %weight_copy = check.file.read.npy path("{directory/'weight.npy'}") : tensor<{wb}xi8>
  %up = check.file.read.npy path("{directory/'up.npy'}") : tensor<{wb}xi8>
  %up_copy = check.file.read.npy path("{directory/'up.npy'}") : tensor<{wb}xi8>
  %expected = check.file.read.npy path("{directory/'expected.npy'}") : tensor<{tp*n}xf32>
  %count = check.file.read.npy path("{directory/'count.npy'}") : tensor<3xi32>
  %count_copy = check.file.read.npy path("{directory/'count.npy'}") : tensor<3xi32>
  %scales = check.generate.fill value(-991.0) : tensor<{tp*2}xf32>
  %half = check.generate.fill value(-991.0) : tensor<{tp*k}xf16>
  %owner = check.generate.fill value(-991.0) : tensor<{tp*n+8}xf32>
  %output = check.tensor.view %owner offset(16) : tensor<{tp*n+8}xf32> -> tensor<{tp*n}xf32>
  kernel.launch @llm.projection.prepare_scaled_f16[%T,%Tp,%K](%T,%Tp,%K,%origin,%count,%input,%scales,%half) : [index,index,index](index,index,index,offset,tensor<3xi32>,tensor<{t*k}xf32>,tensor<{tp*2}xf32>,tensor<{tp*k}xf16>)
"""
            text += (
                "  kernel.launch @llm.platform."
                + kernel
                + "[%Tp,%K,%N](%Tp,%K,%N,%count,%half,%scales,%weight,"
                + ("%up," if gated else "")
                + "%output) : [index,index,index](index,index,index,tensor<3xi32>,tensor<"
                + str(tp * k)
                + "xf16>,tensor<"
                + str(tp * 2)
                + "xf32>,tensor<"
                + str(wb)
                + "xi8>,"
                + ("tensor<" + str(wb) + "xi8>," if gated else "")
                + "tensor<"
                + str(tp * n)
                + "xf32>)\n"
            )
            text += f"""  check.expect.approximate actual(%output) expected(%expected) atol({atol}) rtol({rtol}) nan(same) : tensor<{tp*n}xf32>, tensor<{tp*n}xf32>
  check.expect.bitwise actual(%input) expected(%input_copy) : tensor<{t*k}xf32>
  check.expect.bitwise actual(%weight) expected(%weight_copy) : tensor<{wb}xi8>
  check.expect.bitwise actual(%up) expected(%up_copy) : tensor<{wb}xi8>
  check.expect.bitwise actual(%count) expected(%count_copy) : tensor<3xi32>
  check.file.write.npy value(%owner) path("actual.npy") mode(always) : tensor<{tp*n+8}xf32>
  check.return
}}
"""
            fixture = directory / "case.loom"
            fixture.write_text(text)
            parse = [
                str(args.loom_link),
                "--mode=merge",
                "--allow-unresolved",
                "--to=bc",
                str(fixture),
                "--output=" + str(directory / "case.loombc"),
            ]
            with (directory / "parse.log").open("w") as log:
                parsed = subprocess.run(
                    parse, stdout=log, stderr=subprocess.STDOUT, timeout=60
                )
            assert parsed.returncode == 0, (directory / "parse.log").read_text()
            first_capture = None
            for library in libraries:
                run = directory / library.suffix[1:]
                run.mkdir()
                cmd = [
                    str(args.runner),
                    str(fixture),
                    "--library=" + str(library),
                    "--case=@real_matrix",
                    "--device=amdgpu",
                    "--file-output-root=" + str(run),
                ]
                start = time.monotonic()
                with (run / "execution.log").open("w") as log:
                    result = subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120
                    )
                record = {
                    "case": name,
                    "capacity": t,
                    "padded_capacity": tp,
                    "active": active,
                    "library": str(library),
                    "library_sha256": sha(library),
                    "command": cmd,
                    "exit": result.returncode,
                    "seconds": time.monotonic() - start,
                    "bindings": bindings,
                }
                if result.returncode == 0:
                    capture = (run / "actual.npy").read_bytes()
                    if first_capture is not None:
                        assert capture == first_capture, "SOURCE/BC output differs"
                    first_capture = capture
                    values = floats(run / "actual.npy")
                    assert len(values) == tp * n + 8
                    guards = values[:4] + values[-4:]
                    actual = values[4:-4]
                    assert guards == [-991.0] * 8
                    assert all(math.isfinite(x) for x in actual)
                    assert all(
                        struct.pack("<f", value) == b"\0\0\0\0"
                        for value in actual[active * n :]
                    ), "inactive result must be positive zero bitwise"
                    diff = [
                        a - b
                        for a, b in zip(actual[: active * n], golden[: active * n])
                    ]
                    denominator = max(
                        math.sqrt(math.fsum(x * x for x in golden[: active * n])), 1e-12
                    )
                    error = math.sqrt(math.fsum(x * x for x in diff)) / denominator
                    record.update(
                        relative_l2=error,
                        max_absolute_error=max(map(abs, diff), default=0.0),
                        limit=0.002 if gated else 0.0005,
                        guards=True,
                        padding_zero=True,
                    )
                    assert error <= record["limit"], record
                records.append(record)
                (args.output / "results.json").write_text(
                    json.dumps(
                        {
                            "cases": records,
                            "sources": {str(p): sha(p) for p in sources},
                        },
                        indent=2,
                    )
                    + "\n"
                )
                if result.returncode:
                    print((run / "execution.log").read_text()[-5000:])
                    return result.returncode
                print(
                    name,
                    library.suffix,
                    "PASS relativeL2",
                    record["relative_l2"],
                    flush=True,
                )
    assert {
        str(p): sha(p) for p in sources
    } == source_hashes, "matrix source changed during execution"
    return 0


if __name__ == "__main__":
    sys.exit(main())
