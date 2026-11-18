# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
"""Finite actual-GGUF epoch equivalence/ownership gate, without a math oracle.

Consume emit_epoch_validation outputs at T8/C64 (Gemma: local window +64). Compare B0/S0 and B2/S7
with the same logical inputs, position and persistent state; then continuation,
reset and no-output. Source/bytecode, full state, active residuals and output
slots must match bitwise. Run only in the coordinator's isolated venv/GPU lane.
The opt-in Qwen specialization mode uses actual T128/C512 geometry, poisoned
unconsumed residual rows, mapped-page boundaries and dependent mixed epochs.
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
helper_path = (
    Path(__file__).resolve().parents[2]
    / "architectures/qwen35/testing/run_block_validation.py"
)
spec = importlib.util.spec_from_file_location("epoch_capture_bytes", helper_path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def gemma_window_witness(module, windows):
    """Check this finite materialization's actual all-layer window consumer.

    This is a representation witness for the maintained constructor, not an IR
    interpreter or a substitute for the separate independent window math oracle.
    """
    text = module.read_text()
    assert len(re.findall(r"kernel.launch @gemma3_attention_rows\[", text)) == len(
        windows
    )
    match = re.search(
        r"^kernel.def @gemma3_attention_rows\b.*?(?=^kernel.def |\Z)", text, re.M | re.S
    )
    assert match, "missing actual Gemma attention consumer"
    body = match.group()
    # Lookup starts with the final layer's fallback and selects each earlier
    # layer in order. Native LINK may fold only its initial no-op selections:
    # 1B ends in a local layer, whereas 12B ends in a global layer.
    assert windows and windows[0] > 0 and 0 in windows
    last = len(windows) - 1
    fallback = windows[last]
    arms = {
        int(i): int(value)
        for i, value in re.findall(r"%c(\d+)_0 = index.constant (\d+) : index", body)
    }
    assert arms, "missing per-layer Gemma window selections"
    first = min(arms)
    assert 0 <= first < last
    assert all(value == fallback for value in windows[:first]), windows[:first]
    assert arms == {i: windows[i] for i in range(first, last)}, arms
    assert f"%v{first}_0 = index.constant {fallback} : index" in body
    for i in range(first, last):
        assert f"%l{i} = index.constant {i} : index" in body
        assert f"%is{i} = index.cmp eq, %layer, %l{i} : index" in body
        assert f"%v{i + 1}_0 = scf.select %is{i}, %c{i}_0, %v{i}_0 : index" in body, (
            i,
            "disconnected layer window selection",
        )
    # Check the complete actual window consumer, including the global fallback,
    # bounded local retention and the causal loop's exclusive upper bound.
    assert f"%windowed = index.cmp ne, %v{last}_0, %c0 : index" in body
    assert (
        f"%requested = scf.select %windowed, %v{last}_0, %context_count : index" in body
    )
    assert "%shorter = index.cmp ult, %requested, %context_count : index" in body
    assert "%retained = scf.select %shorter, %requested, %context_count : index" in body
    assert "%key_begin = index.sub %context_count, %retained : index" in body
    assert re.search(
        r"scf.for %token(?:\$\d+)? = \[%key_begin to %context_count step %c1\]", body
    ), "window retention does not feed actual attention traversal"
    return {
        "module_sha256": h.digest(module.read_bytes()),
        "attention_kernel_sha256": h.digest(body.encode()),
        "actual_attention_launches": len(windows),
        "effective_windows": windows,
        "explicit_window_layers": list(range(first, last)),
        "folded_prefix_layers": list(range(first)),
        "fallback_layer": last,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("emitted", "gguf", "reflector", "runner", "output", "gpu-lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--specialized-qwen",
        action="store_true",
        help="actual T128/C512 whole-state specialization cases",
    )
    parser.add_argument(
        "--decode-root",
        default="",
        help="candidate one-row root; omit for original-reference run",
    )
    parser.add_argument(
        "--source-module",
        type=Path,
        help="explicit original SOURCE product with emitted layout",
    )
    parser.add_argument(
        "--bytecode-module",
        type=Path,
        help="explicit original bytecode product with emitted layout",
    )
    args = parser.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert sys.byteorder == "little"
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    args.output.mkdir(parents=True, exist_ok=True)
    layout = json.loads((args.emitted / "layout.json").read_text())
    family = layout["family"]
    assert family in ("llama", "qwen35", "gemma3")
    qwen = family == "qwen35"
    gemma = family == "gemma3"
    context = layout["context"]
    local_window = max(layout["layer_windows"]) if gemma else 0
    assert context % 64 == 0
    tokens_capacity = layout["tokens"]
    if args.specialized_qwen:
        assert qwen and tokens_capacity == 128 and context == 512
        assert not args.decode_root or args.decode_root == layout["decode_root"]
    else:
        assert tokens_capacity == 8 and context == (local_window + 64 if gemma else 64)
        assert not args.decode_root
    hidden = layout["hidden"]
    facts = {
        key: int(value)
        for key, value in re.findall(
            r"config.def @([\w.]+) = (\d+) :",
            (args.emitted / "configuration.loom").read_text(),
        )
    }
    if qwen:
        state_begin = facts["qwen.model.gdn_slot_origin"]
        kv_begin = facts["qwen.model.private_row_stride"]
        state_bytes = kv_begin + context * facts["qwen.model.target_kv_token_bytes"]
        residual_bytes = facts["qwen.model.hidden_capacity_bytes"]
        metadata_bytes = 4192
        map_origin = facts["qwen.model.map_origin"]
        eos = 0  # Qwen numerical publication returns the raw selected token.
        vocabulary = facts["qwen.model.vocabulary_size"]
        plane_count = facts["qwen.model.target_kv_planes"]
        plane_token_bytes = facts["qwen.model.kv_plane_token_bytes"]
        gdn_end = state_begin + facts["qwen.model.gdn_state_bytes"]
    else:
        prefix = "gemma3" if gemma else "llama"
        state_begin = kv_begin = facts[prefix + ".attention.cache_base"]
        state_bytes = (
            kv_begin
            + 4
            * facts[prefix + ".model.layer_count"]
            * context
            * facts[prefix + ".model.kv_size"]
        )
        residual_bytes = 8 * hidden * 4
        metadata_bytes = 2064
        map_origin = facts[prefix + ".attention.map_origin"]
        eos = facts[prefix + ".publication.end_turn_id"]
        vocabulary = facts[prefix + ".model.vocabulary_size"]
        plane_count = 2 * facts[prefix + ".model.layer_count"]
        plane_token_bytes = 2 * facts[prefix + ".model.kv_size"]
        gdn_end = state_begin
    assert vocabulary > 400
    page_count = context // 64
    page_map = [
        ((page + 3) % page_count if args.specialized_qwen else page)
        for page in range(page_count)
    ]
    rows = bytearray([83]) * (map_origin + 4 * page_count + 64)
    struct.pack_into(
        "<4q",
        rows,
        0,
        state_begin,
        state_begin,
        kv_begin,
        0 if args.specialized_qwen else 11,
    )
    for page, physical in enumerate(page_map):
        struct.pack_into("<i", rows, map_origin + page * 4, physical)
    initial_state = (
        bytes([91]) * state_begin + bytes(state_bytes - state_begin) + bytes([91]) * 256
    )
    root = (
        "generated.qwen_epoch"
        if qwen
        else "generated.gemma3_epoch" if gemma else "generated.llama_epoch"
    )
    env = dict(os.environ, PYTHONNOUSERSITE="1", PARAMETER_GGUF=str(args.gguf))
    record = {
        "passed": False,
        "layout": layout,
        "criteria": "bitwise equivalence/ownership; no numerical accuracy qualification",
        "runs": [],
        "comparisons": [],
    }
    prior_form = {}
    if gemma:
        layer_count = facts["gemma3.model.layer_count"]
        global_factor = layout["rope_scaling_factor"]
        assert math.isfinite(global_factor) and global_factor > 0
        factor_match = re.search(
            r"config.def @gemma3.model.rope_factor_global = ([^ ]+) : f32",
            (args.emitted / "configuration.loom").read_text(),
        )
        assert factor_match and math.isclose(
            float(factor_match[1]), global_factor, rel_tol=1e-7
        )
        assert len(layout["layer_windows"]) == layer_count
        assert local_window > 0 and 0 in layout["layer_windows"]
        assert all(window in (0, local_window) for window in layout["layer_windows"])
        assert layout["layer_rope_bases"] == [
            10000 if window else 1000000 for window in layout["layer_windows"]
        ]
        assert plane_count == 2 * layer_count
        assert (
            plane_token_bytes
            == 2 * facts["gemma3.model.kv_heads"] * facts["gemma3.model.head_size"]
        )
        assert state_bytes == kv_begin + plane_count * context * plane_token_bytes
        record["window_validation"] = {
            "native_route": gemma_window_witness(
                args.emitted / "epoch.loom", layout["layer_windows"]
            ),
            "configuration_sha256": h.digest(
                (args.emitted / "configuration.loom").read_bytes()
            ),
            "layer_windows": layout["layer_windows"],
            "layer_rope_bases": layout["layer_rope_bases"],
            "local_rotary_factor": 1,
            "global_rotary_factor": global_factor,
            "numerical_oracle": "separate shared attention window oracle; this gate proves full-model routing and ownership",
            "criteria": f"P{local_window - 1}/N3 then dependent P{local_window + 2}/N1; excluded local KV invariance and included lower-edge sensitivity",
        }

    def publish():
        (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")

    for form in ("loom", "loombc"):
        module = (
            args.source_module if form == "loom" else args.bytecode_module
        ) or args.emitted / ("epoch." + form)
        work = args.output / form
        work.mkdir(exist_ok=True)
        with (
            (work / "parameters.tsv").open("w") as out,
            (work / "reflection.log").open("w") as err,
        ):
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
        fd = os.memfd_create("actual-epoch-immutable-bank", os.MFD_CLOEXEC)
        header = h.npy_header(length)
        os.ftruncate(fd, len(header) + length)
        os.pwrite(fd, header, 0)
        payloads = []
        with args.gguf.open("rb", buffering=0) as source:
            for line in (work / "parameters.tsv").read_text().splitlines():
                name, offset, size, buffer, origin = line.split("\t")
                assert int(buffer) == 0
                source.seek(int(origin))
                remaining, destination = int(size), len(header) + int(offset)
                while remaining:
                    data = source.read(min(remaining, 1024 * 1024))
                    assert data and os.pwrite(fd, data, destination) == len(data)
                    remaining -= len(data)
                    destination += len(data)
                payloads.append(
                    {"name": name, "bytes": int(size), "source": int(origin)}
                )
        assert payloads
        record["runs"].append(
            {
                "form": form,
                "module_sha256": h.digest(module.read_bytes()),
                "bindings": payloads,
            }
        )

        # Independently reflect the second public root before reusing a bank.
        # A different layout must be handled explicitly, never fed normal-root bytes.
        if args.decode_root:
            with (
                (work / "decode-parameters.tsv").open("w") as out,
                (work / "decode-reflection.log").open("w") as err,
            ):
                subprocess.run(
                    [str(args.reflector), args.decode_root, str(module)],
                    env=env,
                    stdout=out,
                    stderr=err,
                    check=True,
                )
            assert (work / "decode-parameters.tsv").read_bytes() == (
                work / "parameters.tsv"
            ).read_bytes(), "public roots require different parameter banks"
            decode_roots = re.findall(
                r"ROOT (\d+) length=(\d+) alignment=(\d+)",
                (work / "decode-reflection.log").read_text(),
            )
            assert decode_roots == roots
        observations = {}

        def run(
            name,
            begin,
            slot,
            position,
            tokens,
            state,
            selected=True,
            poison_word=0x7FC50001,
        ):
            case = work / name
            case.mkdir(exist_ok=True)
            epoch = bytearray([79]) * (metadata_bytes + 64)
            count = int(selected)
            struct.pack_into(
                "<8i",
                epoch,
                0,
                begin + len(tokens),
                1,
                count,
                len(tokens),
                position,
                begin,
                0,
                slot if selected else -1,
            )
            struct.pack_into("<i", epoch, 1292, begin + len(tokens) - 1)
            struct.pack_into("<i", epoch, 1292 + 7 * 4, 0)
            struct.pack_into("<2i", epoch, 1548, 0, 0)
            struct.pack_into("<i", epoch, 2060, eos)
            # Qwen eagerly embeds valid padding. Llama's stronger active guard
            # must avoid invalid IDs outside its descriptor's input span.
            ids = [0 if qwen else -1] * tokens_capacity
            ids[begin : begin + len(tokens)] = tokens
            input_ids = (
                struct.pack("<" + "i" * tokens_capacity, *ids) + bytes([71]) * 64
            )
            output_ids = struct.pack("<20i", *(-99 - 7 * i for i in range(20)))
            residual = struct.pack("<f", -31.0) * (residual_bytes // 4 + 64)
            case_root = (
                args.decode_root
                if args.decode_root and begin == 0 and len(tokens) == 1
                else root
            )
            specialized = case_root != root
            if specialized:
                # Only the candidate's forbidden-read suffix is poisoned. The
                # original embedding legitimately writes all static ID rows.
                residual = (
                    residual[: hidden * 4]
                    + struct.pack("<I", poison_word) * (residual_bytes // 4 - hidden)
                    + residual[residual_bytes:]
                )
            buffers = [
                residual,
                bytes(epoch),
                bytes(rows),
                state,
                input_ids,
                output_ids,
            ]
            for i, data in enumerate(buffers, 1):
                h.save(case / f"arg{i}.npy", data)
            declarations = ",".join(f"%a{i}:buffer" for i in range(7))
            tensors = ",".join(
                [f"tensor<{length}xi8>"] + [f"tensor<{len(x)}xi8>" for x in buffers]
            )
            fixture = f"command.decl @{case_root}() launch({declarations})\nfunc.def @noop({declarations}) {{ func.return }}\ncheck.scenario @actual_epoch {{\n check.trial[1](%trial:index,%entropy:check.entropy) {{\n"
            fixture += f'  %a0 = check.file.read.npy path("/proc/self/fd/{fd}") : tensor<{length}xi8>\n'
            for i, data in enumerate(buffers, 1):
                fixture += f'  %a{i} = check.file.read.npy path("{case / f"arg{i}.npy"}") : tensor<{len(data)}xi8>\n'
            fixture += f"  check.compare<@{case_root},@noop>(%a0,%a1,%a2,%a3,%a4,%a5,%a6) : ({tensors}) -> () {{\n"
            for i in (0, 2, 3, 5):
                size = length if i == 0 else len(buffers[i - 1])
                fixture += f"   check.expect.bitwise actual(%a{i}) expected(%a{i}) : tensor<{size}xi8>\n"
            fixture += "  }\n }\n check.return\n}\n"
            (case / "check.loom").write_text(fixture)
            captures = case / "capture"
            captures.mkdir(exist_ok=True)
            command = [
                str(args.runner),
                str(case / "check.loom"),
                "--library=" + str(module),
                "--case=@actual_epoch",
                "--device=amdgpu",
            ]
            with (
                args.gpu_lock.open("a") as lock,
                (case / "execution.log").open("w") as log,
            ):
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(
                    command,
                    env=dict(env, LOOM_FFN_CAPTURE_DIR=str(captures)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(fd,),
                )
            assert result.returncode == 0, case / "execution.log"
            actual = {
                i: h.load(captures / f"product0_call0_after_arg{i}.npy")
                for i in range(1, 7)
            }
            assert actual[1][residual_bytes:] == residual[residual_bytes:]
            if specialized:
                assert (
                    actual[1][hidden * 4 : residual_bytes]
                    == residual[hidden * 4 : residual_bytes]
                ), (name, "specialized root touched unconsumed residual suffix")
            assert actual[4][:state_begin] == state[:state_begin]
            assert actual[4][state_bytes:] == state[state_bytes:]
            assert actual[4] != state, (
                form,
                name,
                "persistent computation must execute",
            )
            permitted = [(state_begin, gdn_end)] if qwen else []
            permitted += [
                (
                    kv_begin
                    + (plane * context + page_map[logical // 64] * 64 + logical % 64)
                    * plane_token_bytes,
                    kv_begin
                    + (
                        plane * context
                        + page_map[logical // 64] * 64
                        + logical % 64
                        + 1
                    )
                    * plane_token_bytes,
                )
                for plane in range(plane_count)
                for logical in range(position, position + len(tokens))
            ]
            cursor = 0
            for low, high in sorted(permitted):
                assert actual[4][cursor:low] == state[cursor:low], (
                    form,
                    name,
                    "unrelated state",
                )
                cursor = high
            assert actual[4][cursor:] == state[cursor:], (
                form,
                name,
                "unrelated state tail",
            )
            assert (
                actual[2] == buffers[1]
                and actual[3] == buffers[2]
                and actual[5] == buffers[4]
            )
            out = bytearray(output_ids)
            if selected:
                out[slot * 4 : slot * 4 + 4] = actual[6][slot * 4 : slot * 4 + 4]
            assert actual[6] == bytes(out), (
                form,
                name,
                "only selected physical output may change",
            )
            active_begin = begin if qwen else 0
            active = actual[1][
                active_begin * hidden * 4 : (active_begin + len(tokens)) * hidden * 4
            ]
            values = h.floats(active)
            assert all(math.isfinite(value) for value in values), (
                form,
                name,
                "nonfinite active residual",
            )
            assert any(x != 0.0 for x in values) and active != residual[: len(active)]
            if gemma:
                active_end = (active_begin + len(tokens)) * hidden * 4
                assert (
                    actual[1][active_end:residual_bytes]
                    == residual[active_end:residual_bytes]
                ), (form, name, "inactive residual")
            selected_id = actual[6][slot * 4 : slot * 4 + 4] if selected else None
            if selected:
                assert 0 <= struct.unpack("<i", selected_id)[0] < vocabulary
            proof = (
                h.digest(active),
                h.digest(actual[4]),
                selected_id.hex() if selected_id else None,
            )
            if gemma:
                observations[name] = (
                    h.digest(active),
                    selected_id,
                    tuple(
                        h.digest(
                            actual[4][
                                kv_begin
                                + (plane * context + position)
                                * plane_token_bytes : kv_begin
                                + (plane * context + position + len(tokens))
                                * plane_token_bytes
                            ]
                        )
                        for plane in range(plane_count)
                    ),
                )
            if form == "loom":
                prior_form[name] = proof
            else:
                assert prior_form[name] == proof, (name, "SOURCE/BC")
            comparison = {
                "form": form,
                "case": name,
                "position": position,
                "begin": begin,
                "length": len(tokens),
                "slot": slot if selected else -1,
                "root": case_root,
                "module_sha256": h.digest(module.read_bytes()),
                "specialized_suffix_preserved": specialized,
                "active_sha256": proof[0],
                "full_state_sha256": proof[1],
                "selected_i32_bytes": proof[2],
            }
            if gemma:
                comparison["input_state_sha256"] = h.digest(state)
                comparison["written_kv_sha256"] = observations[name][2]
            record["comparisons"].append(comparison)
            publish()
            for path in captures.glob("*.npy"):
                path.unlink()
            for path in case.glob("arg*.npy"):
                path.unlink()
            return proof, actual[4]

        if args.specialized_qwen:

            def prediction(proof):
                assert proof[2] is not None
                return struct.unpack("<i", bytes.fromhex(proof[2]))[0]

            full, full_state = run(
                "full_prefill", 0, 0, 0, [100 + i for i in range(128)], initial_state
            )
            d1, state = run(
                "decode_after_full", 0, 7, 128, [prediction(full)], full_state
            )
            d2, state = run("second_decode", 0, 0, 129, [prediction(d1)], state)
            partial, state = run(
                "partial_prefill_after_decode",
                0,
                0,
                130,
                [200 + i for i in range(17)],
                state,
            )
            d3, state = run(
                "decode_after_partial", 0, 0, 147, [prediction(partial)], state
            )
            _, state = run(
                "no_output_decode", 0, 0, 148, [prediction(d3)], state, False
            )
            run("decode_after_no_output", 0, 0, 149, [401], state)
            # Independent fresh chains cross a real64-row physical page boundary.
            prefix, prefix_state = run(
                "page_prefix", 0, 0, 0, [100 + i for i in range(63)], initial_state
            )
            edge, edge_state = run(
                "decode_page63", 0, 0, 63, [prediction(prefix)], prefix_state
            )
            across, _ = run("decode_page64", 0, 0, 64, [prediction(edge)], edge_state)
            poison_a, _ = run(
                "suffix_poison_a",
                0,
                7,
                128,
                [prediction(full)],
                full_state,
                poison_word=0x7FC50001,
            )
            poison_b, _ = run(
                "suffix_poison_b",
                0,
                7,
                128,
                [prediction(full)],
                full_state,
                poison_word=0x7FC60002,
            )
            assert poison_a == poison_b
            fresh, _ = run(
                "reset_replay", 0, 0, 0, [100 + i for i in range(128)], initial_state
            )
            assert fresh == full
            os.close(fd)
            continue

        canonical, canonical_state = run(
            "canonical", 0, 0, 11, [100, 200, 300], initial_state
        )
        offset, offset_state = run("offset", 2, 7, 11, [100, 200, 300], initial_state)
        assert canonical == offset, (form, "B0/S0 vs B2/S7")
        continuation, _ = run(
            "canonical_continuation", 0, 0, 14, [400], canonical_state
        )
        offset_continuation, _ = run(
            "offset_continuation", 2, 7, 14, [400], offset_state
        )
        assert continuation == offset_continuation, (form, "dependent continuation")
        reset, _ = run("reset", 2, 7, 11, [100, 200, 300], initial_state)
        assert reset == offset, (form, "reset replay")
        no_output, _ = run("no_output", 2, 7, 11, [100, 200, 300], initial_state, False)
        assert no_output[:2] == offset[:2], (form, "unselected computation/state")
        if gemma:
            # Finite, fully defined prior F16 K/V. This is a legal synthetic cache,
            # not a claim that the prefix was produced by a reference tokenizer/model.
            window_position = local_window - 1
            decode_position = window_position + 3
            history = bytearray(initial_state)
            width = plane_token_bytes // 2
            for plane in range(plane_count):
                for position in range(window_position):
                    values = [
                        (((plane * 7 + position * 3 + lane) % 31) - 15) / 64.0
                        for lane in range(width)
                    ]
                    offset = kv_begin + (plane * context + position) * plane_token_bytes
                    history[offset : offset + plane_token_bytes] = struct.pack(
                        "<" + "e" * width, *values
                    )
            history = bytes(history)
            window, window_state = run(
                "window_prefill", 0, 0, window_position, [100, 200, 300], history
            )
            window_offset, window_offset_state = run(
                "window_offset", 2, 7, window_position, [100, 200, 300], history
            )
            assert window == window_offset, (form, "window crossing B/S equivalence")
            window_decode, _ = run(
                "window_decode", 0, 0, decode_position, [400], window_state
            )
            window_decode_offset, _ = run(
                "window_offset_decode",
                2,
                7,
                decode_position,
                [400],
                window_offset_state,
            )
            assert window_decode == window_decode_offset, (
                form,
                "window dependent decode",
            )
            excluded = bytearray(window_state)
            for layer, window_size in enumerate(layout["layer_windows"]):
                if not window_size:
                    continue
                for plane in (2 * layer, 2 * layer + 1):
                    for position in range(
                        3
                    ):  # P=W+2 includes [3, W+2], excluding [0, 2].
                        offset = (
                            kv_begin + (plane * context + position) * plane_token_bytes
                        )
                        excluded[offset : offset + plane_token_bytes] = struct.pack(
                            "<" + "e" * width, *([4.0] * width)
                        )
            assert bytes(excluded) != window_state
            run(
                "window_excluded_local_prefix",
                0,
                0,
                decode_position,
                [400],
                bytes(excluded),
            )
            assert (
                observations["window_excluded_local_prefix"]
                == observations["window_decode"]
            ), (form, "excluded local prefix changed computation")
            # Position3 is the included lower edge. Change only the first local
            # layer's V row so full-model sensitivity has one unambiguous cause.
            included = bytearray(window_state)
            offset = kv_begin + (context + 3) * plane_token_bytes
            included[offset : offset + plane_token_bytes] = struct.pack(
                "<" + "e" * width, *([8.0] * width)
            )
            run(
                "window_included_local_edge",
                0,
                0,
                decode_position,
                [400],
                bytes(included),
            )
            assert (
                observations["window_included_local_edge"][0]
                != observations["window_decode"][0]
            ), (form, "included local lower edge had no effect")
            global_prefix = bytearray(window_state)
            first_global = layout["layer_windows"].index(0)
            plane = 2 * first_global + 1
            for position in range(3):
                offset = kv_begin + (plane * context + position) * plane_token_bytes
                global_prefix[offset : offset + plane_token_bytes] = struct.pack(
                    "<" + "e" * width, *([8.0] * width)
                )
            run(
                "window_included_global_prefix",
                0,
                0,
                decode_position,
                [400],
                bytes(global_prefix),
            )
            assert (
                observations["window_included_global_prefix"][0]
                != observations["window_decode"][0]
            ), (form, "global attention incorrectly discarded prefix")
            record["window_validation"].setdefault("metamorphic_checks", []).append(
                {
                    "form": form,
                    "excluded_local_prefix_unchanged": True,
                    "included_local_lower_edge_sensitive": True,
                    "global_prefix_sensitive": True,
                }
            )
            publish()
        os.close(fd)
    record["passed"] = True
    publish()


if __name__ == "__main__":
    main()
