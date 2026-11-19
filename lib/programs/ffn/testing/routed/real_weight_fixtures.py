#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Selected actual-weight witnesses, never a full-model accuracy claim.

Usage: real_weight_fixtures.py HELPERS.py FACTS.json OUTPUT_DIR DOWN_TAG
All inputs/budgets are fixed before GPU evaluation. Outputs must be external.
"""

import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path

BUDGETS = {
    "logits": (1e-5, 1e-4),
    "probabilities": (2e-5, 1e-4),
    "weights": (2e-5, 1e-4),
    "gate_values": (1e-4, 5e-4),
    "up_values": (1e-4, 5e-4),
    "hidden": (5e-4, 2e-3),
    "down_values": (1e-3, 3e-3),
    "combined": (1e-3, 3e-3),
    "residual": (1e-3, 3e-3),
    "head": (1e-4, 5e-4),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_range(values):
    require(values and all(math.isfinite(v) for v in values), "nonfinite/empty oracle")
    return {"min": min(values), "max": max(values), "max_abs": max(map(abs, values))}


class Actual:
    def __init__(self, helpers, facts, tag):
        self.h = helpers
        self.facts = facts
        require(tag in (12, 14), "DOWN_TAG must be 12 or 14")
        require(facts["architecture"] == "qwen3moe", "expected qwen3moe architecture")
        self.source = Path(facts["source_asset"])
        self.stat = self.source.stat()
        require(
            self.stat.st_size == facts["file_size"], "source size differs from facts"
        )
        self.records = facts["tensors"]
        self.ranges = []
        self.pack_reports = []
        self.tag = tag
        candidates = []
        for name, record in self.records.items():
            require(record["name"] == name, "tensor key/name mismatch")
            for key in ("tag", "offset", "bytes", "rank"):
                require(type(record[key]) is int, f"noninteger {name}.{key}")
            require(
                len(record["dims"]) == 4
                and all(type(v) is int and v > 0 for v in record["dims"]),
                f"invalid dimensions: {name}",
            )
            require(
                1 <= record["rank"] <= 4
                and record["offset"] >= 0
                and record["bytes"] > 0
                and record["offset"] + record["bytes"] <= self.stat.st_size,
                f"invalid range/rank: {name}",
            )
            match = re.fullmatch(r"blk\.(\d+)\.ffn_down_exps\.weight", name)
            if match and record["tag"] == tag:
                candidates.append(int(match[1]))
        require(candidates, f"no actual layer with expert_down tag {tag}")
        self.layer = min(candidates)
        prefix = f"blk.{self.layer}."
        self.norm = self.record(prefix + "ffn_norm.weight", 0, [2048], "norm")
        self.router = self.record(
            prefix + "ffn_gate_inp.weight", 0, [2048, 128], "router"
        )
        self.gate = self.record(
            prefix + "ffn_gate_exps.weight", 12, [2048, 768, 128], "gate"
        )
        self.up = self.record(prefix + "ffn_up_exps.weight", 12, [2048, 768, 128], "up")
        self.down = self.record(
            prefix + "ffn_down_exps.weight", tag, [768, 2048, 128], "down"
        )
        self.output_norm = self.record("output_norm.weight", 0, [2048], "output_norm")
        output = self.records["output.weight"]
        require(output["tag"] in (8, 14), "actual output must be Q8_0 or Q6_K")
        self.vocabulary = output["dims"][1]
        require(self.vocabulary == 151936, "expected first-profile vocabulary151936")
        self.output = self.record(
            "output.weight", output["tag"], [2048, self.vocabulary], "output"
        )
        self.head_tag = output["tag"]

    def record(self, name, tag, dims, role):
        record = self.records[name]
        require(
            record["tag"] == tag
            and record["rank"] == len(dims)
            and record["dims"] == dims + [1] * (4 - len(dims)),
            f"unsupported {role} name/rank/shape/tag: {name}",
        )
        block, size = (1, 4) if tag == 0 else self.h.BLOCKS[tag]
        require(
            dims[0] % block == 0 and record["bytes"] == math.prod(dims) // block * size,
            f"invalid encoded extent: {name}",
        )
        return record

    def read(self, record, relative, size, row_ids):
        require(
            type(relative) is int
            and type(size) is int
            and relative >= 0
            and size > 0
            and relative + size <= record["bytes"],
            "requested range outside tensor",
        )
        with self.source.open("rb") as source:
            source.seek(record["offset"] + relative)
            data = source.read(size)
        require(len(data) == size, "short actual-weight read")
        self.ranges.append(
            {
                "name": record["name"],
                "relative_offset": relative,
                "offset": record["offset"] + relative,
                "bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
                "row_ids": row_ids,
            }
        )
        return data

    def floats(self, record):
        data = self.read(record, 0, record["bytes"], "all")
        values = list(struct.unpack(f"<{len(data) // 4}f", data))
        numeric_range(values)
        return values, data

    def quantize(self, values):
        codes, metadata = self.h.quantize(values, enforce_fixture_domain=False)
        require(
            len(codes) == len(values) and all(-127 <= q <= 127 for q in codes),
            "invalid quantized activation codes",
        )
        require(
            all(
                math.isfinite(scale)
                and (scale > 0 or not any(codes[i * 32 : i * 32 + 32]))
                for i, (scale, _) in enumerate(metadata)
            ),
            "nonrepresentable activation metadata",
        )
        self.pack_reports.append(
            {
                "values": numeric_range(values),
                "codes": numeric_range(codes),
                "half_scales": numeric_range([scale for scale, _ in metadata]),
            }
        )
        return codes, metadata

    def dot(self, data, tag, packed):
        # Same maintained codec decoder/quantizer as E1; actual bytes replace
        # its synthetic packet builder. Decode grouping remains mathematical.
        codes, metadata = packed
        block, size = self.h.BLOCKS[tag]
        require(len(data) == len(codes) // block * size, "dot row byte mismatch")
        parts = []
        for start in range(0, len(codes), 32):
            packet = data[(start // block) * size : (start // block + 1) * size]
            products = []
            for lane in range(32):
                code, scale, minimum = self.h.decode(packet, tag, start % block + lane)
                require(
                    math.isfinite(scale) and math.isfinite(minimum),
                    "nonfinite weight scale/minimum",
                )
                products.append(codes[start + lane] * (scale * code - minimum))
            parts.append(metadata[start // 32][0] * math.fsum(products))
        value = self.h.F(math.fsum(parts))
        require(math.isfinite(value), "nonfinite actual dot")
        return value


def rms(h, values, gamma):
    total = h.F(0)
    for value in values:
        total = h.F(total + h.F(value * value))
    inverse = h.F(1 / math.sqrt(h.F(h.F(total / len(values)) + h.F(1e-6))))
    result = [h.F(h.F(value * inverse) * scale) for value, scale in zip(values, gamma)]
    numeric_range(result)
    return result


def oracle(actual):
    h = actual.h
    residual = [h.F(0.001 * ((ch % 17) - 8)) for ch in range(2048)]
    gamma, norm_bytes = actual.floats(actual.norm)
    router, router_bytes = actual.floats(actual.router)
    normalized = rms(h, residual, gamma)
    logits = [
        h.F(math.fsum(normalized[ch] * router[e * 2048 + ch] for ch in range(2048)))
        for e in range(128)
    ]
    exponentials = [math.exp(value - max(logits)) for value in logits]
    probabilities = [h.F(value / math.fsum(exponentials)) for value in exponentials]
    order = sorted(range(128), key=lambda e: (-probabilities[e], e))
    margins = []
    for left, right in zip(order[:8], order[1:9]):
        gap = logits[left] - logits[right]
        bound = sum(1e-5 + 1e-4 * abs(logits[e]) for e in (left, right))
        require(
            gap > bound,
            f"fixed-input top8 ill-conditioned: experts {left}/{right}, logit gap {gap}, required > {bound}",
        )
        margins.append({"left": left, "right": right, "gap": gap, "required_gt": bound})
    ids = order[:8]
    denominator = max(sum(probabilities[e] for e in ids), 2**-14)
    weights = [h.F(probabilities[e] / denominator) for e in ids]
    packed_input = actual.quantize(normalized)
    gate_values, up_values, hidden, down_values = [], [], [], []
    gate_chunks, up_chunks, down_chunks = [], [], []
    row_bytes = 2048 // 256 * 144
    down_row_bytes = 768 // h.BLOCKS[actual.tag][0] * h.BLOCKS[actual.tag][1]
    for expert in ids:
        origin = expert * 768 * row_bytes
        gate_data = actual.read(
            actual.gate, origin, 768 * row_bytes, [expert, "all768"]
        )
        up_data = actual.read(actual.up, origin, 768 * row_bytes, [expert, "all768"])
        gate_chunks.append((origin, gate_data))
        up_chunks.append((origin, up_data))
        g = [
            actual.dot(
                gate_data[ch * row_bytes : (ch + 1) * row_bytes], 12, packed_input
            )
            for ch in range(768)
        ]
        u = [
            actual.dot(up_data[ch * row_bytes : (ch + 1) * row_bytes], 12, packed_input)
            for ch in range(768)
        ]
        # Stable scalar SiLU, followed by one F32 hidden rounding.
        values = [
            h.F(
                (
                    gate / (1 + math.exp(-gate))
                    if gate >= 0
                    else gate * math.exp(gate) / (1 + math.exp(gate))
                )
                * up
            )
            for gate, up in zip(g, u)
        ]
        numeric_range(values)
        packed_hidden = actual.quantize(values)
        down = []
        for ch in (0, 1, 2047):
            relative = (expert * 2048 + ch) * down_row_bytes
            data = actual.read(actual.down, relative, down_row_bytes, [expert, ch])
            down_chunks.append((relative, data))
            down.append(actual.dot(data, actual.tag, packed_hidden))
        gate_values.append(g)
        up_values.append(u)
        hidden.append(values)
        down_values.append(down)
    combined = []
    for ch in range(3):
        value = h.F(0)
        for index in range(8):
            value = h.fmadd(weights[index], down_values[index][ch], value)
        combined.append(value)
    # Projection-only conditioning: d=1/128, inverse128 and codes exactly
    # {-127,0,127} in every32 group under both native dense pack policies.
    # This avoids equating MMVQ reciprocal with expert division arithmetic.
    head_input = [h.F(((ch % 3) - 1) * (127 / 128)) for ch in range(2048)]
    packed_head = actual.quantize(head_input)
    require(
        set(packed_head[0]) == {-127, 0, 127}
        and all(scale == 1 / 128 for scale, _ in packed_head[1]),
        "head pack conditioning changed",
    )
    head_row_bytes = 2048 // h.BLOCKS[actual.head_tag][0] * h.BLOCKS[actual.head_tag][1]
    head_chunks, head_values = [], []
    for ch in (0, 1, actual.vocabulary - 1):
        relative = ch * head_row_bytes
        data = actual.read(actual.output, relative, head_row_bytes, [ch])
        head_chunks.append((relative, data))
        head_values.append(actual.dot(data, actual.head_tag, packed_head))
    final = list(residual)
    for ch, value in zip((0, 1, 2047), combined):
        final[ch] = h.F(final[ch] + value)
    return dict(
        residual=residual,
        gamma=gamma,
        norm_bytes=norm_bytes,
        normalized=normalized,
        router_bytes=router_bytes,
        logits=logits,
        probabilities=probabilities,
        ids=ids,
        weights=weights,
        gate_values=gate_values,
        up_values=up_values,
        hidden=hidden,
        down_values=down_values,
        combined=combined,
        final=final,
        head_input=head_input,
        head_values=head_values,
        gate_chunks=gate_chunks,
        up_chunks=up_chunks,
        down_chunks=down_chunks,
        head_chunks=head_chunks,
        margin=probabilities[order[7]] - probabilities[order[8]],
        logit_margin=logits[order[7]] - logits[order[8]],
        margins=margins,
    )


def blob_initializer(out, symbol, chunks):
    # Bounded rodata chunks, vector16 copies and a byte tail, never a constant
    # per weight byte. Offsets retain the full original expert/row strides.
    names = []
    for target, data in chunks:
        for start in range(0, len(data), 65536):
            chunk = data[start : start + 65536]
            name = f"{symbol}.bytes{len(names)}"
            out.add(f'global.rodata.def @{name} = bytes("{chunk.hex()}")')
            names.append((name, target + start, len(chunk)))
    out.func(symbol, ["destination"])
    for index, (name, target, size) in enumerate(names):
        out.add(f"""  %source{index} = global.load @{name} : buffer
  %offset{index} = index.constant {target} : offset
  %src{index} = buffer.view %source{index}[%origin] : buffer -> view<{size}xi8>
  %dst{index} = buffer.view %destination[%offset{index}] : buffer -> view<{size}xi8>
  %zero{index} = index.constant 0 : index
  %step{index} = index.constant 16 : index
  %limit{index} = index.constant {size // 16 * 16} : index
  scf.for %i{index} = [%zero{index} to %limit{index} step %step{index}] {{
    %packet{index} = vector.load %src{index}[%i{index}] : view<{size}xi8> -> vector<16xi8>
    vector.store %packet{index}, %dst{index}[%i{index}] : vector<16xi8>, view<{size}xi8>
  }}""")
        if size % 16:
            out.add(f"""  %end{index} = index.constant {size} : index
  %one{index} = index.constant 1 : index
  scf.for %tail{index} = [%limit{index} to %end{index} step %one{index}] {{
    %byte{index} = view.load %src{index}[%tail{index}] : view<{size}xi8> -> i8
    view.store %byte{index}, %dst{index}[%tail{index}] : i8, view<{size}xi8>
  }}""")
    out.end()


def values(out, key, entries, size, typ="f32"):
    out.view(key, size, typ)
    for index, value in entries:
        out.put(key, index, value, size, typ)


def scenario(h, out, name, tensors, outputs, selected=None, exact=None):
    selected, exact = selected or {}, exact or {}
    generated = h.Emit()
    generated.lines = []
    generated.scenario(
        name,
        tensors,
        outputs,
        approx={key: BUDGETS[key] for key in outputs if key in BUDGETS},
    )
    sizes = {key: size for key, size, _, _ in tensors}
    for line in generated.lines:
        if line.startswith("    check.compare<"):
            for key, indices in selected.items():
                for offset in indices:
                    out.add(
                        f"    %{key}_selected{offset} = check.tensor.view %{key} offset({offset * 4}) : tensor<{sizes[key]}xf32> -> tensor<1xf32>"
                    )
            for key, spans in exact.items():
                for begin, length in spans:
                    out.add(
                        f"    %{key}_exact{begin} = check.tensor.view %{key} offset({begin * 4}) : tensor<{sizes[key]}xf32> -> tensor<{length}xf32>"
                    )
        replaced = False
        for key, indices in selected.items():
            if f"check.expect.approximate actual(%{key}) " in line:
                atol, rtol = BUDGETS[key]
                for offset in indices:
                    out.add(
                        f"      check.expect.approximate actual(%{key}_selected{offset}) expected(%{key}_selected{offset}) atol({atol}) rtol({rtol}) nan(same) : tensor<1xf32>, tensor<1xf32>"
                    )
                replaced = True
        if not replaced:
            out.lines.append(line)
        for key, spans in exact.items():
            if f"check.expect.approximate actual(%{key}) " in line:
                for begin, length in spans:
                    out.add(
                        f"      check.expect.bitwise actual(%{key}_exact{begin}) expected(%{key}_exact{begin}) : tensor<{length}xf32>"
                    )


PACK = "kernel.decl @llm.experts.pack_split_q8(%R:index,%K:index,%M:index) launch(%R0:index,%K0:index,%M0:index,%count:buffer,%x:buffer,%packed:buffer)"
ROUTE = "kernel.decl @routed.observe_route(%C:index) launch(%C0:index,%x:buffer,%router:buffer,%count:buffer,%ids:buffer,%weights:buffer,%logits:buffer,%probabilities:buffer)"
OBSERVE = "kernel.decl @routed.observe_gate_up(%N:index,%R:index) launch(%N0:index,%R0:index,%count:buffer,%ids:buffer,%packed:buffer,%gate:buffer,%up:buffer,%hidden:buffer,%gate_values:buffer,%up_values:buffer)"
DOWN = "template.decl @llm.experts.indexed_matmul(%R:index,%K:index,%N:index,%X:index,%T:index,%M:index,%tag:index,%bytes:index,%count:buffer,%ids:buffer,%packed:buffer,%weight:view<[%bytes]xi8>,%out:buffer)"
COMBINE = "kernel.decl @llm.routing.weighted_top8(%C:index) launch(%C0:index,%count:buffer,%ys:buffer,%weights:buffer,%out:buffer)"
RMS = "template.decl @llm.ffn.rms_routed(%C:index,%E:index,%eps:f32,%GT:index,%UT:index,%DT:index,%GB:index,%UB:index,%DB:index,%residual:buffer,%count:buffer,%gamma:view<[%E]xf32>,%router:view<128x2048xf32>,%gate:view<[%GB]xi8>,%up:view<[%UB]xi8>,%down:view<[%DB]xi8>)"
HEAD = "template.decl @llm.projection.policy_rows(%profile:index,%rows:index,%E:index,%V:index,%tag:index,%bytes:index,%count:buffer,%x:buffer,%weight:view<[%bytes]xi8>,%out:buffer)"


def initialize(h, out, name, tensors, scalar_inputs, blobs):
    # Keep initialization's VM frame bounded: thousands of independent scalar
    # constants/stores retain registers after lowering. Copy the same typed
    # input bytes through the existing chunked rodata loader instead.
    input_blobs = {}
    for key, data in scalar_inputs.items():
        typ = next(typ for k, _, typ, _ in tensors if k == key)
        require(typ in ("f32", "i32"), "unsupported scalar input type")
        code = "f" if typ == "f32" else "i"
        input_blobs[key] = [(0, struct.pack("<" + code * len(data), *data))]
    require(not (input_blobs.keys() & blobs.keys()), "duplicate input initializer")
    blobs = input_blobs | blobs
    for key, chunks in blobs.items():
        blob_initializer(out, name + ".load_" + key, chunks)
    out.func(name + ".init", [key for key, _, _, _ in tensors])
    h.count_init(out, 1)
    for key in blobs:
        out.add(f"  func.call @{name}.load_{key}(%{key}) : (buffer)")
    out.end()


def emit_router(a, o):
    h, out = a.h, a.h.Emit()
    out.add(ROUTE)
    tensors = [
        ("x", 2048, "f32", "0.0"),
        ("router", 262144, "f32", "0.0"),
        ("count", 3, "i32", 0),
        ("ids", 8, "i32", 99),
        ("weights", 8, "f32", "99.0"),
        ("logits", 128, "f32", "99.0"),
        ("probabilities", 128, "f32", "99.0"),
    ]
    name = "real_router"
    initialize(
        h,
        out,
        name,
        tensors,
        {"x": o["normalized"]},
        {"router": [(0, o["router_bytes"])]},
    )
    args = [key for key, _, _, _ in tensors]
    out.command(
        name + ".actual",
        args,
        "  %C = index.constant 1 : index\n  kernel.launch @routed.observe_route[%C](%C,%x,%router,%count,%ids,%weights,%logits,%probabilities) : [index](index,buffer,buffer,buffer,buffer,buffer,buffer,buffer)",
    )
    out.func(name + ".expected", args)
    for key in ("ids", "weights", "logits", "probabilities"):
        values(
            out, key, enumerate(o[key]), len(o[key]), "i32" if key == "ids" else "f32"
        )
    out.end()
    scenario(h, out, name, tensors, ["ids", "weights", "logits", "probabilities"])
    return "".join(out.lines)


def weight_tensors(a):
    return [
        (key, record["bytes"], "i8", 0)
        for key, record in (("gate", a.gate), ("up", a.up), ("down", a.down))
    ]


def emit_experts(a, o):
    h, out = a.h, a.h.Emit()
    for declaration in (PACK, OBSERVE, DOWN, COMBINE):
        out.add(declaration)
    tensors = (
        [
            ("x", 2048, "f32", "0.0"),
            ("ids", 8, "i32", 0),
            ("weights", 8, "f32", "0.0"),
            ("count", 3, "i32", 0),
        ]
        + weight_tensors(a)
        + [
            (key, size, "f32", "99.0")
            for key, size in (
                ("gate_values", 6144),
                ("up_values", 6144),
                ("hidden", 6144),
                ("down_values", 16384),
                ("combined", 2048),
            )
        ]
    )
    name = "real_experts"
    initialize(
        h,
        out,
        name,
        tensors,
        {key: o[key] for key in ("ids", "weights")} | {"x": o["normalized"]},
        {key: o[key + "_chunks"] for key in ("gate", "up", "down")},
    )
    args = [key for key, _, _, _ in tensors]
    out.command(
        name + ".actual",
        args,
        f"""  %C = index.constant 1 : index
  %R = index.constant 8 : index
  %E = index.constant 2048 : index
  %F = index.constant 768 : index
  %X = index.constant 128 : index
  %tag = index.constant {a.tag} : index
  %DB = index.constant {a.down['bytes']} : index
  %origin = index.constant 0 : offset
  %pb = index.constant 2560 : offset
  %hb = index.constant 7680 : offset
  %packed = buffer.alloca<global> align(256) %pb : buffer
  %packed_hidden = buffer.alloca<global> align(256) %hb : buffer
  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>
  kernel.launch @llm.experts.pack_split_q8[%C,%E,%C](%C,%E,%C,%count,%x,%packed) : [index,index,index](index,index,index,buffer,buffer,buffer)
  kernel.launch @routed.observe_gate_up[%F,%R](%F,%R,%count,%ids,%packed,%gate,%up,%hidden,%gate_values,%up_values) : [index,index](index,index,buffer,buffer,buffer,buffer,buffer,buffer,buffer,buffer)
  kernel.launch @llm.experts.pack_split_q8[%R,%F,%R](%R,%F,%R,%count,%hidden,%packed_hidden) : [index,index,index](index,index,index,buffer,buffer,buffer)
  template.apply<@llm.experts.indexed_matmul>(%R,%F,%E,%X,%R,%C,%tag,%DB,%count,%ids,%packed_hidden,%dv,%down_values) : (index,index,index,index,index,index,index,index,buffer,buffer,buffer,view<[%DB]xi8>,buffer)
  kernel.launch @llm.routing.weighted_top8[%C](%C,%count,%down_values,%weights,%combined) : [index](index,buffer,buffer,buffer,buffer)""",
    )
    out.func(name + ".expected", args)
    selected = {}
    for key in ("gate_values", "up_values", "hidden"):
        entries = [
            (expert * 768 + ch, o[key][expert][ch])
            for expert in range(8)
            for ch in (0, 1, 767)
        ]
        values(out, key, entries, 6144)
        selected[key] = [index for index, _ in entries]
    entries = [
        (expert * 2048 + ch, o["down_values"][expert][j])
        for expert in range(8)
        for j, ch in enumerate((0, 1, 2047))
    ]
    values(out, "down_values", entries, 16384)
    selected["down_values"] = [index for index, _ in entries]
    out.zero("combined", 2048)
    for ch, value in zip((0, 1, 2047), o["combined"]):
        out.put("combined", ch, value, 2048)
    out.end()
    scenario(
        h,
        out,
        name,
        tensors,
        ["gate_values", "up_values", "hidden", "down_values", "combined"],
        selected,
        {"combined": [(2, 2045)]},
    )
    return "".join(out.lines)


def emit_composition(a, o):
    h, out = a.h, a.h.Emit()
    out.add(RMS)
    tensors = [
        ("residual", 2048, "f32", "77.0"),
        ("gamma", 2048, "f32", "0.0"),
        ("router", 262144, "f32", "0.0"),
        ("count", 3, "i32", 0),
    ] + weight_tensors(a)
    name = "real_composition"
    initialize(
        h,
        out,
        name,
        tensors,
        {"residual": o["residual"], "gamma": o["gamma"]},
        {"router": [(0, o["router_bytes"])]}
        | {key: o[key + "_chunks"] for key in ("gate", "up", "down")},
    )
    args = [key for key, _, _, _ in tensors]
    out.command(
        name + ".actual",
        args,
        f"""  %C = index.constant 1 : index
  %E = index.constant 2048 : index
  %eps = scalar.constant {h.F(1e-6)!r} : f32
  %q4 = index.constant 12 : index
  %dt = index.constant {a.tag} : index
  %GB = index.constant {a.gate['bytes']} : index
  %DB = index.constant {a.down['bytes']} : index
  %origin = index.constant 0 : offset
  %nv = buffer.view %gamma[%origin] : buffer -> view<[%E]xf32>
  %rv = buffer.view %router[%origin] : buffer -> view<128x2048xf32>
  %gv = buffer.view %gate[%origin] : buffer -> view<[%GB]xi8>
  %uv = buffer.view %up[%origin] : buffer -> view<[%GB]xi8>
  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>
  template.apply<@llm.ffn.rms_routed>(%C,%E,%eps,%q4,%q4,%dt,%GB,%GB,%DB,%residual,%count,%nv,%rv,%gv,%uv,%dv) : (index,index,f32,index,index,index,index,index,index,buffer,buffer,view<[%E]xf32>,view<128x2048xf32>,view<[%GB]xi8>,view<[%GB]xi8>,view<[%DB]xi8>)""",
    )
    out.func(name + ".expected", args)
    values(out, "residual", enumerate(o["final"]), 2048)
    out.end()
    scenario(h, out, name, tensors, ["residual"], exact={"residual": [(2, 2045)]})
    # Reuse the same immutable rodata and production invocation for inactive
    # and invalid counts; every residual bit must remain the initialized value.
    arguments = ",".join("%" + key for key in args)
    types = ",".join("buffer" for _ in args)
    for active in (0, -1, 2):
        inactive = name + ("_negative" if active < 0 else f"_count{active}")
        out.func(inactive + ".init", args)
        out.add(f"  func.call @{name}.init({arguments}) : ({types})")
        h.count_init(out, active)
        out.end()
        out.command(
            inactive + ".actual",
            args,
            f"  command.launch @{name}.actual({arguments}) : ({types})",
        )
        out.func(inactive + ".expected", args)
        out.end()
        scenario(
            h, out, inactive, tensors, ["residual"], exact={"residual": [(0, 2048)]}
        )
    return "".join(out.lines)


def emit_head(a, o):
    h, out = a.h, a.h.Emit()
    out.add(HEAD)
    tensors = [
        ("x", 2048, "f32", "0.0"),
        ("weight", a.output["bytes"], "i8", 0),
        ("count", 3, "i32", 0),
        ("head", a.vocabulary, "f32", "99.0"),
    ]
    name = "real_head"
    initialize(
        h, out, name, tensors, {"x": o["head_input"]}, {"weight": o["head_chunks"]}
    )
    args = [key for key, _, _, _ in tensors]
    out.command(
        name + ".actual",
        args,
        f"""  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %E = index.constant 2048 : index
  %V = index.constant {a.vocabulary} : index
  %tag = index.constant {a.head_tag} : index
  %bytes = index.constant {a.output['bytes']} : index
  %origin = index.constant 0 : offset
  %wv = buffer.view %weight[%origin] : buffer -> view<[%bytes]xi8>
  template.apply<@llm.projection.policy_rows>(%zero,%one,%E,%V,%tag,%bytes,%count,%x,%wv,%head) : (index,index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)""",
    )
    out.func(name + ".expected", args)
    out.zero("head", a.vocabulary)
    for ch, value in zip((0, 1, a.vocabulary - 1), o["head_values"]):
        out.put("head", ch, value, a.vocabulary)
    out.end()
    scenario(h, out, name, tensors, ["head"], exact={"head": [(2, a.vocabulary - 3)]})
    return "".join(out.lines)


def main():
    require(len(sys.argv) == 5, __doc__)
    helper_path, facts_path, destination, tag = sys.argv[1:]
    output = Path(destination).resolve()
    require(not output.exists(), "OUTPUT_DIR must be fresh and external")
    repository = Path(__file__).resolve().parents[5]
    require(
        not output.is_relative_to(repository), "OUTPUT_DIR must be outside repository"
    )
    spec = importlib.util.spec_from_file_location("actual_weight_helpers", helper_path)
    require(
        spec is not None and spec.loader is not None,
        "cannot load explicit helper module",
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    facts_bytes = Path(facts_path).read_bytes()
    actual = Actual(helpers, json.loads(facts_bytes), int(tag))
    observed = oracle(actual)
    final_stat = actual.source.stat()
    require(
        all(
            getattr(final_stat, key) == getattr(actual.stat, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "immutable source changed during reads",
    )
    report = {
        "source_asset": str(actual.source),
        "file_size": actual.stat.st_size,
        "facts_sha256": hashlib.sha256(facts_bytes).hexdigest(),
        "layer": actual.layer,
        "down_tag": actual.tag,
        "head_tag": actual.head_tag,
        "selected_experts": observed["ids"],
        "selected_weights": observed["weights"],
        "top8_probability_margin": observed["margin"],
        "top8_logit_margin": observed["logit_margin"],
        "budgets": BUDGETS,
        "ranges": actual.ranges,
        "coverage": "selected8 full gate/up experts, down channels0/1/2047, output rows0/1/V-1; not full-model accuracy",
        "numeric_ranges": {
            key: numeric_range(observed[key])
            for key in (
                "residual",
                "normalized",
                "gamma",
                "logits",
                "probabilities",
                "weights",
                "combined",
                "final",
                "head_input",
                "head_values",
            )
        },
    }
    for key in ("gate_values", "up_values", "hidden", "down_values"):
        report["numeric_ranges"][key] = numeric_range(
            [v for row in observed[key] for v in row]
        )
    report["top8_adjacent_logit_conditioning"] = observed["margins"]
    require(
        len(actual.pack_reports) == 10, "expected normalized/8hidden/head pack reports"
    )
    report["pack_rows_normalized_then_selected_hidden_then_head"] = actual.pack_reports
    report["head_conditioning"] = (
        "projection-only fixed ((ch%3)-1)*127/128, each32group d=1/128 inverse128 codes{-127,0,127}; finalnorm/greedy are separate full-model gates"
    )
    report["input_f32_sha256"] = {
        key: hashlib.sha256(
            struct.pack(f"<{len(observed[key])}f", *observed[key])
        ).hexdigest()
        for key in ("residual", "normalized", "head_input")
    }
    report["rms_epsilon_f32"] = helpers.F(1e-6)
    report["selected_tensors"] = {
        record["name"]: record
        for record in (
            actual.norm,
            actual.router,
            actual.gate,
            actual.up,
            actual.down,
            actual.output,
        )
    }
    report["encoded_bytes_read"] = sum(entry["bytes"] for entry in actual.ranges)
    report["gpu_weight_bytes_per_expert_or_composition_command"] = (
        actual.gate["bytes"] + actual.up["bytes"] + actual.down["bytes"]
    )
    report["pack_domain"] = {
        "synthetic_amplitude_bound_enforced": False,
        "all_scales_finite_half_representable": True,
        "normalized_max_abs": max(map(abs, observed["normalized"])),
        "hidden_row_max_abs": [max(map(abs, row)) for row in observed["hidden"]],
        "head_input_max_abs": max(map(abs, observed["head_input"])),
    }
    # Generation and all numerical assertions precede any public-check outputs.
    modules = {
        "router.loom": emit_router(actual, observed),
        "experts.loom": emit_experts(actual, observed),
        "composition.loom": emit_composition(actual, observed),
        "head.loom": emit_head(actual, observed),
    }
    output.mkdir(parents=True)
    for name, text in modules.items():
        (output / name).write_text(text)
    (output / "oracle.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        AssertionError,
        OverflowError,
        struct.error,
    ) as error:
        raise SystemExit(f"real_weight_fixtures: {error}") from error
