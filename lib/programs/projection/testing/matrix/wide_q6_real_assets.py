#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Full-shape immutable Qwen32 down witness with bounded scalar expectations.

Invoked by real_assets.py --wide-q6-down; no full-model emulator or model export.
"""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

import canonical_fixtures as canonical_helpers
from canonical_fixtures import f32
from fixtures import stage_weight
from real_assets import write_npy, floats
from wide_q6_fixtures import canonical

QWEN32_SHA256 = "2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e"


def launches(text, constants, buffers):
    return [
        [
            symbol,
            [constants[x.strip()] for x in grid.split(",")],
            [
                constants[x.strip()] if x.strip() in constants else buffers[x.strip()]
                for x in arguments.split(",")
            ],
        ]
        for symbol, grid, arguments in re.findall(
            r"kernel.launch @([\w.]+)\[([^]]+)\]\(([^)]+)\)", text
        )
    ]


def lowered_matrix_proof(path, capacity):
    """Bounded association with the actual strict-normalized matrix root."""
    text = path.read_text()
    root = text.split("\n}", 1)[0]
    assert re.search(
        r"command.def public retain @real_down_c" + str(capacity) + r"\(\)", root
    )
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[\w$]+) = index.constant (\d+) : (?:index|offset)", root
        )
    }
    buffers = {
        "%count": "count",
        "%input": "input",
        "%weight": "weight",
        "%output": "output",
    }
    scratch = {}
    for name, size in re.findall(
        r"(%[\w$]+) = buffer.alloca<global> align\(256\) (%[\w$]+)", root
    ):
        size = constants[size]
        role = {31104: "packed", 110592: "defined", 3538944: "half", 512: "scales"}[
            size
        ]
        assert role not in scratch
        scratch[role] = size
        buffers[name] = role
    for name, source, origin in re.findall(
        r"(%[\w$]+) = buffer.view (%[\w$]+)\[(%[\w$]+)\]", root
    ):
        assert constants[origin] == 0 and buffers[source] == "weight"
        buffers[name] = "weight"
    actual = launches(root, constants, buffers)
    assert scratch == (
        {"packed": 31104, "defined": 110592}
        if capacity == 1
        else {"half": 3538944, "scales": 512}
    )
    configurations = {}
    for symbol, _, _ in actual:
        match = re.search(
            r"kernel.def[^\n]* @"
            + re.escape(symbol)
            + r"\([^\n]*\) \{\n(.*?)\n\} launch",
            text,
            re.S,
        )
        assert match, symbol
        configurations[symbol] = match.group(0).split("} launch", 1)[0]
    return dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        launches=actual,
        scratch=scratch,
        configurations=configurations,
        targets=re.findall(r"amdgpu.target[^\n]+", text),
    )


def run(args):
    assert (
        args.provider_text and args.provider_bc
    ), "supply coherent full operation products"
    assert args.matrix_proof_root, "supply strict-normalized C1/C17 matrix proofs"
    proofs = {
        (capacity, form): [
            lowered_matrix_proof(
                args.matrix_proof_root
                / f"{target}-{form}-c{capacity}"
                / "normalized.loom",
                capacity,
            )
            for target in ("gfx1100", "gfx1151")
        ]
        for capacity in (1, 17)
        for form in ("text", "bc")
    }
    strict_records = json.loads(
        args.matrix_proof_root.with_name(
            args.matrix_proof_root.name + "-results.json"
        ).read_text()
    )
    for capacity in (1, 17):
        for form, library in (("text", args.provider_text), ("bc", args.provider_bc)):
            identity = hashlib.sha256(library.read_bytes()).hexdigest()
            for target in ("gfx1100", "gfx1151"):
                record = next(
                    x
                    for x in strict_records
                    if x["capacity"] == capacity
                    and x["form"] == form
                    and x["profile"] == "amdgpu:" + target
                )
                assert (
                    record["exit"] == 0
                    and record["K"] == 27648
                    and record["N"] == 5120
                    and record["provider_sha256"] == identity
                )
    assert not args.output.exists(), "use a fresh output directory"
    # Hash streaming, never load the checkpoint into memory or modify it.
    digest = hashlib.sha256()
    with args.gguf.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == QWEN32_SHA256, "wrong audited Qwen32 checkpoint"
    index = json.loads(args.tensor_index.read_text())
    tensor = next(x for x in index["tensors"] if x["name"] == "blk.0.ffn_down.weight")
    assert tensor["dimensions"] == [27648, 5120] and tensor["storage_type"] == 14
    k, n = tensor["dimensions"]
    row_bytes = k // 256 * 210
    size, offset = n * row_bytes, tensor["absolute_offset"]
    assert offset >= 0 and offset + size <= args.gguf.stat().st_size
    # Two immutable weight views, source/copy, output and bounded scratch remain
    # comfortably below this fixed witness allocation ceiling. No whole model.
    allocation_upper_bound = (
        2 * size + 2 * 17 * k * 4 + 64 * k * 2 + 64 * n * 4 + 1048576
    )
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert allocation_upper_bound < 512 * 1024 * 1024
    assert (
        available > 2 * allocation_upper_bound
    ), "insufficient host allocation headroom"
    if "bytes" in tensor:
        assert tensor["bytes"] == size
    with args.gguf.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(size)
    assert len(raw) == size
    args.output.mkdir(parents=True)
    write_npy(args.output / "weight.npy", raw, "|i1", size)
    binding = dict(
        tensor=tensor["name"],
        dimensions=[k, n],
        tag=14,
        offset=offset,
        bytes=size,
        end=offset + size,
        sha256=hashlib.sha256(raw).hexdigest(),
        gguf_sha256=digest.hexdigest(),
        allocation_upper_bound=allocation_upper_bound,
    )
    # Fixed before execution. Probe low/high and last columns; all K blocks are
    # decoded independently for these columns. Each provider keeps its own oracle.
    selected = (0, 1, 7, 8, 63, 127, 1023, 2559, 4095, 5119)
    source = [f32((((j * 17 + j // 32 * 3) % 255) - 127) / 128) for j in range(k)]
    source[0] = 127 / 128
    packed = canonical(canonical_helpers, source)
    oracles = {}
    for provider in ("canonical", "scaled"):
        inp = packed if provider == "canonical" else source
        values = []
        for column in selected:
            base = column * row_bytes
            decoded = []
            for block in range(k // 256):
                wire = raw[base + block * 210 : base + (block + 1) * 210]
                for j in range(256):
                    if provider == "scaled":
                        w = stage_weight(wire, 14, j)
                    else:
                        half, within = divmod(j, 128)
                        quarter, lane = divmod(within, 32)
                        low = (
                            wire[half * 64 + quarter % 2 * 32 + lane]
                            >> (4 * (quarter // 2))
                        ) & 15
                        high = (wire[128 + half * 32 + lane] >> (2 * quarter)) & 3
                        scale = struct.unpack_from("<b", wire, 192 + j // 16)[0]
                        d = struct.unpack_from("<e", wire, 208)[0]
                        w = f32(f32(d * scale) * ((low | high << 4) - 32))
                    decoded.append(w)
            values.append(f32(math.fsum(x * w for x, w in zip(inp, decoded))))
        oracles[provider] = values
    records = []
    for capacity in (1, 17):
        padded = capacity if capacity == 1 else 64
        provider = "canonical" if capacity == 1 else "scaled"
        for active in (1, 0, 1):
            ordinal = len(records) // 2
            directory = args.output / f"c{capacity}-step{ordinal}-a{active}"
            directory.mkdir()
            inputs = source + [math.nan] * ((capacity - 1) * k)
            if active == 0:
                inputs = [math.nan] * (capacity * k)
            write_npy(
                directory / "input.npy",
                struct.pack("<" + "f" * len(inputs), *inputs),
                "<f4",
                len(inputs),
            )
            expected = oracles[provider] if active else [0.0] * len(selected)
            text = f"""template.decl @llm.projection.matrix(%C:index,%K:index,%N:index,%tag:index,%B:index,%count:buffer,%input:buffer,%weight:view<[%B]xi8>,%output:buffer)
kernel.decl @llm.projection.prepare_f32_prefix(%C:index,%K:index) launch(%C:index,%K:index,%count:buffer,%input:buffer,%output:buffer)
kernel.decl @ggml_quantize_q8_1_x4_f32(%C:index,%K:index) launch(%C:index,%K:index,%input:buffer,%output:buffer)
kernel.decl @ggml_linear_q6k_q8_1_x4(%C:index,%K:index,%N:index) launch(%C:index,%K:index,%N:index,%packed:buffer,%weight:buffer,%output:buffer)
kernel.decl @llm.projection.prepare_scaled_f16(%C:index,%P:index,%K:index) launch(%C:index,%P:index,%K:index,%origin:offset,%count:buffer,%input:buffer,%scales:buffer,%output:buffer)
kernel.decl @llm.platform.q6k_scaled_f16_store(%P:index,%K:index,%N:index) launch(%P:index,%K:index,%N:index,%count:buffer,%input:buffer,%scales:buffer,%weight:buffer,%output:buffer)
"""
            for fmt in (
                "q4k",
                "q5k",
                "q6k",
                "q8_0",
                "q5_0",
                "q4_0",
                "q4_1",
                "iq4_xs",
                "iq4_nl",
                "iq3_s",
                "q3k",
            ):
                text += f"config.def @ggml.linear_{fmt}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{fmt}_q8_1_x4.output_capacity = 262144 : index\n"
            text += f"""config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index
command.def public @real_down_c{capacity}() launch(%count:buffer,%input:buffer,%weight:buffer,%output:buffer) {{
  %C = index.constant {capacity} : index
  %K = index.constant {k} : index
  %N = index.constant {n} : index
  %B = index.constant {size} : index
  %tag = index.constant 14 : index
  %o = index.constant 0 : offset
  %w = buffer.view %weight[%o] : buffer -> view<[%B]xi8>
  template.apply<@llm.projection.matrix>(%C,%K,%N,%tag,%B,%count,%input,%w,%output) : (index,index,index,index,index,buffer,buffer,view<[%B]xi8>,buffer)
  command.return
}}
// The operation root above retains strict matrix materialization. The standard
// pinned check.case planner supports kernel launches, not command.launch. Run
// its selected launch chain using the maintained real_assets.py case pattern.
check.case public @wide_q6_real_down {{
  %C = check.literal value({capacity}) : index
  %P = check.literal value({padded}) : index
  %K = check.literal value({k}) : index
  %N = check.literal value({n}) : index
  %origin = check.literal value(0) : offset
  %count = check.generate.fill value({active}) : tensor<3xi32>
  %input = check.file.read.npy path("{directory/'input.npy'}") : tensor<{capacity*k}xf32>
  %input_copy = check.file.read.npy path("{directory/'input.npy'}") : tensor<{capacity*k}xf32>
  %weight = check.file.read.npy path("{args.output/'weight.npy'}") : tensor<{size}xi8>
  %weight_copy = check.file.read.npy path("{args.output/'weight.npy'}") : tensor<{size}xi8>
  %count_copy = check.generate.fill value({active}) : tensor<3xi32>
  %owner = check.generate.fill value(-991.0) : tensor<{padded*n+8}xf32>
  %output = check.tensor.view %owner offset(16) : tensor<{padded*n+8}xf32> -> tensor<{padded*n}xf32>
"""
            if capacity == 1:
                packed_bytes = capacity * k // 128 * 144
                text += f"""  %defined = check.generate.fill value(-991.0) : tensor<{capacity*k}xf32>
  %packed = check.generate.fill value(-23) : tensor<{packed_bytes}xi8>
  kernel.launch @llm.projection.prepare_f32_prefix[%C,%K](%C,%K,%count,%input,%defined) : [index,index](index,index,tensor<3xi32>,tensor<{capacity*k}xf32>,tensor<{capacity*k}xf32>)
  kernel.launch @ggml_quantize_q8_1_x4_f32[%C,%K](%C,%K,%defined,%packed) : [index,index](index,index,tensor<{capacity*k}xf32>,tensor<{packed_bytes}xi8>)
  kernel.launch @ggml_linear_q6k_q8_1_x4[%C,%K,%N](%C,%K,%N,%packed,%weight,%output) : [index,index,index](index,index,index,tensor<{packed_bytes}xi8>,tensor<{size}xi8>,tensor<{padded*n}xf32>)
"""
            else:
                text += f"""  %scales = check.generate.fill value(-991.0) : tensor<{padded*2}xf32>
  %half = check.generate.fill value(-991.0) : tensor<{padded*k}xf16>
  kernel.launch @llm.projection.prepare_scaled_f16[%C,%P,%K](%C,%P,%K,%origin,%count,%input,%scales,%half) : [index,index,index](index,index,index,offset,tensor<3xi32>,tensor<{capacity*k}xf32>,tensor<{padded*2}xf32>,tensor<{padded*k}xf16>)
  kernel.launch @llm.platform.q6k_scaled_f16_store[%P,%K,%N](%P,%K,%N,%count,%half,%scales,%weight,%output) : [index,index,index](index,index,index,tensor<3xi32>,tensor<{padded*k}xf16>,tensor<{padded*2}xf32>,tensor<{size}xi8>,tensor<{padded*n}xf32>)
"""
            text += f"""
  check.expect.bitwise actual(%input) expected(%input_copy) : tensor<{capacity*k}xf32>
  check.expect.bitwise actual(%weight) expected(%weight_copy) : tensor<{size}xi8>
  check.expect.bitwise actual(%count) expected(%count_copy) : tensor<3xi32>
  check.file.write.npy value(%owner) path("actual.npy") mode(always) : tensor<{padded*n+8}xf32>
  check.return
}}
"""
            fixture = directory / "case.loom"
            case_body = text.split("check.case public @wide_q6_real_down", 1)[1]
            case_constants = {
                name: int(value)
                for name, value in re.findall(
                    r"(%[\w$]+) = check.literal value\((\d+)\) : (?:index|offset)",
                    case_body,
                )
            }
            case_buffers = {
                "%" + name: name
                for name in (
                    "count",
                    "input",
                    "weight",
                    "output",
                    "defined",
                    "packed",
                    "half",
                    "scales",
                )
            }
            executed = launches(case_body, case_constants, case_buffers)
            for form in ("text", "bc"):
                for proof in proofs[(capacity, form)]:
                    assert executed == proof["launches"], (executed, proof)
                    for role, byte_size in proof["scratch"].items():
                        match = re.search(
                            "%"
                            + role
                            + r" = check.generate.fill[^\n]+tensor<(\d+)x(f32|f16|i8)>",
                            case_body,
                        )
                        assert (
                            match
                            and int(match[1]) * {"f32": 4, "f16": 2, "i8": 1}[match[2]]
                            == byte_size
                        )
            fixture.write_text(text)
            captures = []
            for fmt, library in (
                ("text", args.provider_text),
                ("bc", args.provider_bc),
            ):
                run_dir = directory / fmt
                run_dir.mkdir()
                cmd = [
                    str(args.runner),
                    str(fixture),
                    "--library=" + str(library),
                    "--case=@wide_q6_real_down",
                    "--device=amdgpu",
                    "--file-output-root=" + str(run_dir),
                ]
                with (
                    args.gpu_lock.open("a") as lock,
                    (run_dir / "execution.log").open("w") as log,
                ):
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, timeout=180
                    )
                record = dict(
                    capacity=capacity,
                    active=active,
                    provider=provider,
                    form=fmt,
                    command=cmd,
                    exit=result.returncode,
                    binding=binding,
                    selected_columns=list(selected),
                    expected=expected,
                    operation_proofs=proofs[(capacity, fmt)],
                    execution="actual lowered matrix kernel chain; no command runtime invocation",
                    provider_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                )
                records.append(record)
                (args.output / "results.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                assert result.returncode == 0, (run_dir / "execution.log").read_text()[
                    -8000:
                ]
                values = floats(run_dir / "actual.npy")
                assert values[:4] + values[-4:] == [-991.0] * 8
                actual = values[4:-4]
                assert all(math.isfinite(x) for x in actual)
                assert all(
                    struct.pack("<f", x) == b"\0\0\0\0" for x in actual[active * n :]
                )
                # Existing fixture accuracy budget; fixed before any GPU output.
                atol, rtol = (
                    (0.0001, 0.0005) if provider == "scaled" else (0.00001, 0.00001)
                )
                samples = [actual[c] for c in selected]
                assert all(
                    abs(a - e) <= atol + rtol * abs(e)
                    for a, e in zip(samples, expected)
                ), (samples, expected)
                captures.append(b"".join(struct.pack("<f", x) for x in actual))
                record.update(
                    actual_samples=samples,
                    atol=atol,
                    rtol=rtol,
                    guards=True,
                    padding_zero=True,
                    input_immutable=True,
                )
                (args.output / "results.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
            assert captures[0] == captures[1], "SOURCE/BC numerical output differs"
    return 0
