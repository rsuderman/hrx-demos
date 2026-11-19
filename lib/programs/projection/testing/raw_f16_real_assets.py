#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Selected actual raw-half columns through the shared prepared operation.

Use the shared emit_tensor_facts catalog output, never another GGUF parser.
This proves sampled full-K dots, not full-logit or whole-model accuracy. Existing
consumed full-width head, layer/state and Serve checks remain separate gates.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import site
import struct
import subprocess
import sys
import time

from f16_rows_fixtures import f32, half_dot


def npy(path, kind, data, count):
    header = repr(dict(descr=kind, fortran_order=False, shape=(count,)))
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + data
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture(rows, width, columns, weight_bytes, paths):
    total = rows * columns
    text = f"""template.decl @llm.projection.project_prepared(%policy:index,%rows:index,%capacity:index,%K:index,%N:index,%tag:index,%B:index,%fixed:i32,%origin:offset,%count:buffer,%input:buffer,%packed:buffer,%weight:view<[%B]xi8>,%output:buffer)
command.def @actual.raw_f16() launch(%input:buffer,%weights:buffer,%count:buffer,%output:buffer,%expected:buffer) {{
  %policy = index.constant 2 : index
  %R = index.constant {rows} : index
  %K = index.constant {width} : index
  %N = index.constant {columns} : index
  %tag = index.constant 1 : index
  %B = index.constant {weight_bytes} : index
  %origin = index.constant 0 : offset
  %dynamic = scalar.constant -1 : i32
  %w = buffer.view %weights[%origin] : buffer -> view<[%B]xi8>
  template.apply<@llm.projection.project_prepared>(%policy,%R,%R,%K,%N,%tag,%B,%dynamic,%origin,%count,%input,%input,%w,%output) : (index,index,index,index,index,index,index,i32,offset,buffer,buffer,buffer,view<[%B]xi8>,buffer)
  command.return
}}
func.def @expected.raw_f16(%input:buffer,%weights:buffer,%count:buffer,%output:buffer,%expected:buffer) {{
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %total = index.constant {total} : index
  %out = buffer.view %output[%origin] : buffer -> view<{total}xf32>
  %ref = buffer.view %expected[%origin] : buffer -> view<{total}xf32>
  scf.for %i = [%zero to %total step %one] {{
    %value = view.load %ref[%i] : view<{total}xf32> -> f32
    view.store %value,%out[%i] : f32,view<{total}xf32>
  }}
  func.return
}}
check.scenario @actual_f16_weights {{
 check.trial[1](%trial:index,%entropy:check.entropy) {{
"""
    for name, count, kind in (
        ("input", rows * width + 8, "f32"),
        ("weights", weight_bytes + 32, "i8"),
        ("count", 3, "i32"),
        ("expected", total, "f32"),
    ):
        path = json.dumps(str(paths[name]))
        owner = name + "_owner" if name in ("input", "weights") else name
        text += (
            f"  %{owner} = check.file.read.npy path({path}) : tensor<{count}x{kind}>\n"
        )
    text += f"""  %input = check.tensor.view %input_owner offset(16) : tensor<{rows*width+8}xf32> -> tensor<{rows*width}xf32>
  %weights = check.tensor.view %weights_owner offset(16) : tensor<{weight_bytes+32}xi8> -> tensor<{weight_bytes}xi8>
  %output_owner = check.generate.fill value(-888.0) : tensor<{total+8}xf32>
  %output = check.tensor.view %output_owner offset(16) : tensor<{total+8}xf32> -> tensor<{total}xf32>
"""
    kinds = (
        f"tensor<{rows*width}xf32>",
        f"tensor<{weight_bytes}xi8>",
        "tensor<3xi32>",
        f"tensor<{total}xf32>",
        f"tensor<{total}xf32>",
    )
    text += (
        "  check.compare<@actual.raw_f16,@expected.raw_f16>(%input,%weights,%count,%output,%expected) : ("
        + ",".join(kinds)
        + ") -> () {\n"
    )
    owners = (
        ("input_owner", f"tensor<{rows*width+8}xf32>"),
        ("weights_owner", f"tensor<{weight_bytes+32}xi8>"),
        ("count", "tensor<3xi32>"),
        ("output_owner", f"tensor<{total+8}xf32>"),
        ("expected", f"tensor<{total}xf32>"),
    )
    for name, kind in owners:
        text += f"   check.expect.bitwise actual(%{name}) expected(%{name}) : {kind}\n"
    return text + "  }\n }\n check.return\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "gguf",
        "tensor-facts",
        "provider-text",
        "provider-bc",
        "runner",
        "output",
        "gpu-lock",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--tensor", action="append", help="repeat actual tensor name")
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    assert os.environ.get("LOOM_SERVE_DIR") and args.output.is_absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    facts = json.loads(args.tensor_facts.read_text())
    assert Path(facts["source_asset"]).resolve() == args.gguf.resolve()
    before = args.gguf.stat()
    assert facts["file_size"] == before.st_size
    tensors = args.tensor or [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "token_embd.weight",
    ]
    assert len(tensors) == len(set(tensors))
    providers = {"source": args.provider_text, "bytecode": args.provider_bc}
    record = dict(
        evidence="sampled actual raw-F16 full-K dots, not full-model accuracy",
        source_asset=str(args.gguf.resolve()),
        tensor_facts_sha256=digest(args.tensor_facts.read_bytes()),
        provider_sha256={k: digest(p.read_bytes()) for k, p in providers.items()},
        oracle_source_sha256=digest(
            Path(__file__).with_name("f16_rows_fixtures.py").read_bytes()
        ),
        oracle="IEEEhalf via struct; original F32, ascending-K F32 multiply/add",
        criterion="bitwise including output guards and immutable inputs",
        passed=False,
        runs=[],
    )

    def publish():
        (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")

    publish()
    with args.gguf.open("rb") as source:
        for ordinal, name in enumerate(tensors):
            tensor = facts["tensors"][name]
            width, full_columns, tail0, tail1 = tensor["dims"]
            assert tensor["rank"] == 2 and tensor["tag"] == 1
            assert tail0 == tail1 == 1 and 0 < width <= 32768 and full_columns > 0
            assert tensor["bytes"] == width * full_columns * 2
            assert 0 <= tensor["offset"] <= before.st_size - tensor["bytes"]
            selected = sorted({0, full_columns // 2, full_columns - 1})
            wires, identities = [], []
            for column in selected:
                offset = tensor["offset"] + column * width * 2
                source.seek(offset)
                wire = source.read(width * 2)
                assert len(wire) == width * 2
                assert all(math.isfinite(v) for (v,) in struct.iter_unpack("<e", wire))
                wires.append(wire)
                identities.append(
                    dict(
                        column=column,
                        absolute_offset=offset,
                        bytes=len(wire),
                        sha256=digest(wire),
                    )
                )
            sampled = b"".join(wires)
            for rows, active in ((1, 1), (8, 3)):
                work = args.output / f"tensor{ordinal}-r{rows}"
                work.mkdir()
                paths = {
                    key: work / (key + ".npy")
                    for key in ("input", "weights", "count", "expected")
                }
                inputs = [
                    [
                        f32((r + 1) * (((k % 17) - 8) / 16 + 1 / 4096))
                        for k in range(width)
                    ]
                    for r in range(active)
                ]
                x = [-777.0] * 4 + [v for row in inputs for v in row]
                x += [float("nan")] * ((rows - active) * width) + [-777.0] * 4
                expected = [
                    half_dot(inputs[r], wire) if r < active else 0.0
                    for r in range(rows)
                    for wire in wires
                ]
                npy(paths["input"], "<f4", struct.pack("<" + "f" * len(x), *x), len(x))
                npy(
                    paths["weights"],
                    "|i1",
                    b"\xe9" * 16 + sampled + b"\xe9" * 16,
                    len(sampled) + 32,
                )
                npy(paths["count"], "<i4", struct.pack("<3i", active, 17, 31), 3)
                npy(
                    paths["expected"],
                    "<f4",
                    struct.pack("<" + "f" * len(expected), *expected),
                    len(expected),
                )
                case = work / "case.loom"
                case.write_text(
                    fixture(rows, width, len(selected), len(sampled), paths)
                )
                try:
                    for form, provider in providers.items():
                        command = [
                            str(args.runner),
                            str(case),
                            "--library=" + str(provider),
                            "--case=@actual_f16_weights",
                            "--device=amdgpu",
                        ]
                        started = time.monotonic()
                        with (
                            args.gpu_lock.open("a") as lock,
                            (work / (form + ".log")).open("w") as log,
                        ):
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            result = subprocess.run(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=False,
                                timeout=180,
                            )
                        record["runs"].append(
                            dict(
                                tensor=name,
                                actual_dimensions=tensor["dims"],
                                selected=identities,
                                rows=rows,
                                active=active,
                                form=form,
                                command=command,
                                exit=result.returncode,
                                fixture_sha256=digest(case.read_bytes()),
                                expected_sha256=digest(paths["expected"].read_bytes()),
                                seconds=time.monotonic() - started,
                            )
                        )
                        publish()
                        assert result.returncode == 0, (name, rows, form)
                finally:
                    for path in paths.values():
                        path.unlink(missing_ok=True)
    after = args.gguf.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    record["generated_payloads_removed"] = True
    record["passed"] = True
    publish()


if __name__ == "__main__":
    main()
