#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Frozen independent finite-domain E1 oracles; never reads GPU output.

Wire decode uses scalar GGML byte indexes; activation packing uses F32 division,
round-away and half-RNE. Sparse rows make exact full-shape operation checks bounded.
The observed providers expose actual intermediate arithmetic, not oracle code.
"""

import ctypes
import hashlib
import math
import re
import struct
import sys
from pathlib import Path

F = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
H = lambda x: struct.unpack("<e", struct.pack("<e", x))[0]
libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
libm.fmaf.restype = ctypes.c_float
fmadd = lambda a, b, c: float(libm.fmaf(a, b, c))
GATE_TAGS = (12, 13, 14, 8)
DOWN_TAGS = GATE_TAGS + (6,)
BLOCKS = {12: (256, 144), 13: (256, 176), 14: (256, 210), 8: (32, 34), 6: (32, 22)}


def weight_bytes(tag):
    values, size = BLOCKS[tag]
    return 128 * 2048 * 768 // values * size


def quantize(values, enforce_fixture_domain=True):
    codes, meta = [], []
    for start in range(0, len(values), 32):
        xs = values[start : start + 32]
        maximum = max(map(abs, xs))
        if enforce_fixture_domain:
            assert maximum == 0 or 2**-6 <= maximum <= 8, maximum
        else:
            assert all(math.isfinite(value) for value in xs), xs
        d = F(maximum / 127)
        if not enforce_fixture_domain and maximum:
            assert math.isfinite(d) and d > 0 and math.isfinite(H(d)) and H(d) > 0, d
        qs = []
        for x in xs:
            q = F(x / d) if maximum else 0
            qs.append(math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5))
        codes += qs
        meta.append((H(d), sum(qs)))
    return codes, meta


def packet(tag, expert, channel, block, up=False, down=False, amplitude=1):
    if tag == 12:
        data = bytearray(144)
        # Exactly representable positive scales, independent gate and up storage.
        d = (1 / 32 if down else 1 / 16 if up else 1 / 8) * (1 + (expert % 2) / 2)
        struct.pack_into("<ee", data, 0, d, d / 4)
        data[4:12] = bytes([1]) * 8
        data[12:16] = bytes([0x11]) * 4
        nibble = 5 + (expert + channel + block + (2 if up else 0)) % 5
        data[16:] = bytes([nibble * 17]) * 128
    elif tag == 14:
        data = bytearray(210)
        data[:128] = bytes([0xA6 + channel % 2]) * 128
        data[128:192] = bytes([0x59]) * 64
        data[192:208] = bytes(
            ((-1 if j % 2 else 1) * (1 + expert % 2)) % 256 for j in range(16)
        )
        struct.pack_into("<e", data, 208, amplitude / 32)
    elif tag == 13:
        data = bytearray(176)
        d = (1 / 64 if down or up else 1 / 32) * (1 + (expert % 2) / 2)
        struct.pack_into("<ee", data, 0, d, d / 4)
        data[4:12] = bytes([1]) * 8
        data[12:16] = bytes([0x11]) * 4
        data[16:48] = bytes([0x55]) * 32
        nibble = 5 + (expert + channel + block + (2 if up else 0)) % 5
        data[48:] = bytes([nibble * 17]) * 128
    elif tag in (8, 6):
        data = bytearray(BLOCKS[tag][1])
        d = (1 / 64 if down or up else 1 / 32) * (1 + (expert % 2) / 2)
        struct.pack_into("<e", data, 0, d)
        code = 5 + (expert + channel + block + (2 if up else 0)) % 5
        codes = [code] * 32
        codes[1], codes[31] = -code, code - 1
        if tag == 8:
            data[2:] = bytes(q % 256 for q in codes)
        else:
            values = [q + 16 for q in codes]
            high = sum((v >> 4) << i for i, v in enumerate(values))
            struct.pack_into("<I", data, 2, high)
            data[6:] = bytes(
                (values[i] & 15) | ((values[i + 16] & 15) << 4) for i in range(16)
            )
    else:
        raise ValueError(tag)
    return bytes(data)


def decode(data, tag, k):
    if tag in (12, 13):
        d, m = struct.unpack_from("<ee", data)
        sc = data[4:16]
        g = k // 32
        if g < 4:
            scale, minimum = sc[g] & 63, sc[g + 4] & 63
        else:
            scale = (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4)
            minimum = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4)
        q = (data[(16 if tag == 12 else 48) + g // 2 * 32 + k % 32] >> (g % 2 * 4)) & 15
        if tag == 13:
            q |= ((data[16 + k % 32] >> g) & 1) << 4
        return q, F(d * scale), F(m * minimum)
    if tag in (8, 6):
        d = struct.unpack_from("<e", data)[0]
        if tag == 8:
            code = struct.unpack_from("<b", data, 2 + k)[0]
        else:
            high = struct.unpack_from("<I", data, 2)[0]
            code = (data[6 + k % 16] >> (4 * (k // 16))) & 15
            code += ((high >> k) & 1) << 4
            code -= 16
        return code, F(d), 0.0
    assert tag == 14
    half, rem = divmod(k, 128)
    quarter, lane = divmod(rem, 32)
    low = (data[half * 64 + quarter % 2 * 32 + lane] >> (quarter // 2 * 4)) & 15
    high = (data[128 + half * 32 + lane] >> (2 * quarter)) & 3
    code = low + 16 * high - 32
    scale = struct.unpack_from("<b", data, 192 + half * 8 + quarter * 2 + lane // 16)[0]
    return code, F(struct.unpack_from("<e", data, 208)[0] * scale), 0.0


def dot(tag, expert, channel, values, up=False, amplitude=1):
    qs, metadata = quantize(values)
    parts = []
    for start in range(0, len(values), 32):
        group = start // 32
        block_values = BLOCKS[tag][0]
        data = packet(
            tag,
            expert,
            channel,
            start // block_values,
            up,
            down=len(values) == 768,
            amplitude=amplitude,
        )
        dx, raw_sum = metadata[group]
        products = []
        for i in range(32):
            code, scale, minimum = decode(data, tag, start % block_values + i)
            products.append(qs[start + i] * (scale * code - minimum))
        # Independent mathematical sum of exactly decoded quantized values.
        parts.append(float(dx) * math.fsum(products))
    return F(math.fsum(parts))


def xrow(row, width=2048):
    xs = [0.0] * width
    factor = (1, 1.125, 1.25)[row % 3]
    for i, v in ((0, 1), (1, -0.5), (2, 0.25), (width - 1, 0.75)):
        xs[i] = F(v * factor)
    return xs


def route(row, mode):
    if mode == 0:
        logits = [0.0] * 128
    elif mode == 1:
        logits = [
            F(
                (1 + 0.25 * (e - 120) if e >= 120 else -3 + e / 128)
                * (1, 1.125, 1.25)[row % 3]
            )
            for e in range(128)
        ]
    else:
        logits = [F((1, 1.125, 1.25)[row % 3])] * 128
    exps = [math.exp(v - max(logits)) for v in logits]
    probabilities = [F(v / math.fsum(exps)) for v in exps]
    ids = sorted(range(128), key=lambda e: (-probabilities[e], e))[:8]
    den = max(sum(probabilities[e] for e in ids), 2**-14)
    weights = [F(probabilities[e] / den) for e in ids]
    if mode == 1:
        assert min(logits) >= -4 and max(logits) <= 4, logits
        assert all(logits[ids[i]] - logits[ids[i + 1]] >= 0.25 for i in range(7))
        assert all(
            probabilities[ids[i]] - probabilities[ids[i + 1]] >= 1e-4 for i in range(7)
        )
    return logits, probabilities, ids, weights


class Emit:
    def __init__(self):
        self.lines = [
            "// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n// Frozen independent finite-domain anchors generated by fixtures.py.\n"
        ]
        self.serial = 0

    def add(self, text):
        self.lines.append(text + "\n")

    def store(self, view, index, value, typ="f32", shape=None):
        n = self.serial
        self.serial += 1
        if typ == "i8":
            value = value if value < 128 else value - 256
        literal = repr(float(value)) if typ.startswith("f") else str(int(value))
        self.add(
            f"  %v{n} = scalar.constant {literal} : {typ}\n  view.store %v{n}, %{view}[{index}] : {typ}, view<{shape or typ}>"
        )

    def func(self, name, args):
        self.add(
            f"func.def @{name}("
            + ", ".join(f"%{a}: buffer" for a in args)
            + ") {\n  %origin = index.constant 0 : offset"
        )

    def end(self, kind="func"):
        self.add(f"  {kind}.return\n}}")

    def zero(self, name, size, typ="f32"):
        uid = self.serial
        self.serial += 1
        self.add(
            f"""  %{name}_view = buffer.view %{name}[%origin] : buffer -> view<{size}x{typ}>
  %zero{uid} = index.constant 0 : index
  %one{uid} = index.constant 1 : index
  %limit{uid} = index.constant {size} : index
  %value{uid} = scalar.constant {0.0 if typ.startswith('f') else 0} : {typ}
  scf.for %i{uid} = [%zero{uid} to %limit{uid} step %one{uid}] {{
    view.store %value{uid}, %{name}_view[%i{uid}] : {typ}, view<{size}x{typ}>
  }}"""
        )

    def view(self, name, size, typ="f32"):
        self.add(
            f"  %{name}_view = buffer.view %{name}[%origin] : buffer -> view<{size}x{typ}>"
        )

    def put(self, name, index, value, size, typ="f32"):
        self.store(name + "_view", index, value, typ, f"{size}x{typ}")

    def command(self, name, args, body):
        self.add(
            f"command.def @{name}() launch("
            + ", ".join(f"%{a}: buffer" for a in args)
            + ") {\n"
            + body
            + "\n  command.return\n}"
        )

    def scenario(self, name, tensors, outputs, init=True, approx=None, repeat=1):
        # tensors: name,size,type,fill; output views always have 16-byte guards.
        self.add(
            f"check.scenario @{name} {{\n  check.trial[{repeat}](%trial: index, %entropy: check.entropy) {{"
        )
        for key, size, typ, fill in tensors:
            if key in outputs:
                self.add(
                    f"""    %{key}_owner = check.generate.fill value({fill}) : tensor<{size + 8}x{typ}>
    %{key} = check.tensor.view %{key}_owner offset(16) : tensor<{size + 8}x{typ}> -> tensor<{size}x{typ}>
    %{key}_prefix = check.tensor.view %{key}_owner offset(0) : tensor<{size + 8}x{typ}> -> tensor<4x{typ}>
    %{key}_suffix = check.tensor.view %{key}_owner offset({4 * (size + 4)}) : tensor<{size + 8}x{typ}> -> tensor<4x{typ}>"""
                )
            else:
                self.add(
                    f"    %{key} = check.generate.fill value({fill}) : tensor<{size}x{typ}>"
                )
        inactive = {}
        match = re.search(r"_c(\d+)_a(negative|\d+)", name)
        if match:
            c = int(match.group(1))
            a = -1 if match.group(2) == "negative" else int(match.group(2))
            active = a if 0 <= a <= c else 0
            for key, size, typ, _ in tensors:
                if key in outputs and key in (approx or {}) and active < c:
                    remaining = (c - active) * (size // c)
                    inactive[key] = (remaining, typ)
                    self.add(
                        f"    %{key}_inactive = check.tensor.view %{key} offset({4 * active * (size // c)}) : tensor<{size}x{typ}> -> tensor<{remaining}x{typ}>"
                    )
        if name == "activation":
            self.add(
                "    %activation_zero = check.tensor.view %out offset(24) : tensor<14xf32> -> tensor<2xf32>"
            )
        vals = ", ".join("%" + t[0] for t in tensors)
        types = ", ".join(f"tensor<{sz}x{typ}>" for _, sz, typ, _ in tensors)
        if init:
            self.add(f"    check.generate<@{name}.init>({vals}) : ({types}) -> ()")
        self.add(
            f"    check.compare<@{name}.actual, @{name}.expected>({vals}) : ({types}) -> () {{"
        )
        for key, size, typ, _ in tensors:
            tolerance = (approx or {}).get(key)
            if tolerance:
                self.add(
                    f"      check.expect.approximate actual(%{key}) expected(%{key}) atol({float(tolerance[0])}) rtol({float(tolerance[1])}) nan(same) : tensor<{size}x{typ}>, tensor<{size}x{typ}>"
                )
            else:
                self.add(
                    f"      check.expect.bitwise actual(%{key}) expected(%{key}) : tensor<{size}x{typ}>"
                )
            if key in outputs:
                for guard in ["prefix", "suffix"]:
                    self.add(
                        f"      check.expect.bitwise actual(%{key}_{guard}) expected(%{key}_{guard}) : tensor<4x{typ}>"
                    )
        if name == "activation":
            self.add(
                "      check.expect.bitwise actual(%activation_zero) expected(%activation_zero) : tensor<2xf32>"
            )
        for key, (remaining, typ) in inactive.items():
            self.add(
                f"      check.expect.bitwise actual(%{key}_inactive) expected(%{key}_inactive) : tensor<{remaining}x{typ}>"
            )
        self.add("    }\n  }\n  check.return\n}")


def count_init(out, active):
    out.view("count", 3, "i32")
    out.put("count", 0, active, 3, "i32")


def emit_router(out):
    out.add(
        "kernel.decl @routed.route_sums(%C: index) launch(%C0: index, %probabilities: buffer, %weights: buffer, %sums: buffer, %flags: buffer)"
    )
    out.add(
        "kernel.decl @routed.observe_route(%C: index) launch(%C0: index, %x: buffer, %w: buffer, %count: buffer, %ids: buffer, %weights: buffer, %logits: buffer, %probabilities: buffer)"
    )
    for c in CAPACITIES:
        for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
            for mode in (0, 1, 2):
                name = f'route_c{c}_a{a if a >= 0 else "negative"}_m{mode}'
                args = [
                    "x",
                    "router",
                    "count",
                    "ids",
                    "weights",
                    "logits",
                    "probabilities",
                    "sums",
                    "flags",
                ]
                out.func(name + ".init", args)
                count_init(out, a)
                out.zero("x", c * 2048)
                out.view("router", 128 * 2048)
                for t in range(c):
                    out.put(
                        "x",
                        t * 2048,
                        0 if mode == 0 else (1, 1.125, 1.25)[t % 3],
                        c * 2048,
                    )
                for e in range(128):
                    w = (
                        0
                        if mode == 0
                        else (
                            (1 + 0.25 * (e - 120) if e >= 120 else -3 + e / 128)
                            if mode == 1
                            else 1
                        )
                    )
                    out.put("router", e * 2048, w, 128 * 2048)
                out.end()
                out.command(
                    name + ".actual",
                    args,
                    f"""  %C = index.constant {c} : index
  kernel.launch @routed.observe_route[%C](%C, %x, %router, %count, %ids, %weights, %logits, %probabilities) : [index](index, buffer, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @routed.route_sums[%C](%C, %probabilities, %weights, %sums, %flags) : [index](index, buffer, buffer, buffer, buffer)""",
                )
                out.func(name + ".expected", args)
                out.view("sums", c * 2)
                out.view("flags", c, "i32")
                for t in range(c):
                    out.put("flags", t, 1, c, "i32")
                    for k in range(2):
                        out.put("sums", t * 2 + k, 1 if 0 <= t < a <= c else 0, c * 2)
                for key, sz, typ in [
                    ("ids", c * 8, "i32"),
                    ("weights", c * 8, "f32"),
                    ("logits", c * 128, "f32"),
                    ("probabilities", c * 128, "f32"),
                ]:
                    out.zero(key, sz, typ)
                for t in range(max(0, a) if 0 <= a <= c else 0):
                    logits, probs, ids, weights = route(t, mode)
                    for k, (eid, weight) in enumerate(zip(ids, weights)):
                        out.put("ids", t * 8 + k, eid, c * 8, "i32")
                        out.put("weights", t * 8 + k, weight, c * 8)
                    for e in range(128):
                        out.put("logits", t * 128 + e, logits[e], c * 128)
                        out.put("probabilities", t * 128 + e, probs[e], c * 128)
                out.end()
                out.scenario(
                    name,
                    [
                        ("x", c * 2048, "f32", "0.0"),
                        ("router", 128 * 2048, "f32", "0.0"),
                        ("count", 3, "i32", 99),
                        ("ids", c * 8, "i32", 123),
                        ("weights", c * 8, "f32", "99.0"),
                        ("logits", c * 128, "f32", "99.0"),
                        ("probabilities", c * 128, "f32", "99.0"),
                        ("sums", c * 2, "f32", "99.0"),
                        ("flags", c, "i32", 99),
                    ],
                    ["ids", "weights", "logits", "probabilities", "sums", "flags"],
                    approx={
                        "weights": (2e-5, 1e-4),
                        "probabilities": (2e-5, 1e-4),
                        "sums": (2e-5, 0),
                    },
                    repeat=3 if c > 1 and a == c and mode == 1 else 1,
                )


def emit_pack(out):
    out.add(
        "kernel.decl @llm.experts.pack_split_q8(%R: index, %K: index, %M: index) launch(%R0: index, %K0: index, %M0: index, %count: buffer, %x: buffer, %packed: buffer)"
    )
    for c in CAPACITIES:
        for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
            for mult, width in ((1, 2048), (8, 768)):
                rows = c * mult
                total = rows * width * 5 // 4
                name = f'pack_c{c}_a{a if a >= 0 else "negative"}_m{mult}'
                args = ["x", "count", "packed"]
                out.func(name + ".init", args)
                count_init(out, a)
                out.zero("x", rows * width)
                for row in range(rows):
                    xs = xrow(row, width)
                    # Halfway anchors with exactly represented F32 divide: max=127,
                    # d=1 violates fixture pack range, so use max=127/32 and d=1/32.
                    xs[0:4] = [127 / 32, 0.5 / 32, -0.5 / 32, -1.5 / 32]
                    for i, x in enumerate(xs):
                        if x:
                            out.put(
                                "x",
                                row * width + i,
                                x if 0 <= row // mult < a <= c else -777.0,
                                rows * width,
                            )
                out.end()
                # Inactive poison must not be read. Use a finite poison since text
                # literal NaN syntax is intentionally not assumed by the generator.
                out.command(
                    name + ".actual",
                    args,
                    f"""  %R = index.constant {rows} : index
  %K = index.constant {width} : index
  %M = index.constant {mult} : index
  kernel.launch @llm.experts.pack_split_q8[%R, %K, %M](%R, %K, %M, %count, %x, %packed) : [index, index, index](index, index, index, buffer, buffer, buffer)""",
                )
                out.func(name + ".expected", args)
                out.zero("packed", total, "i8")
                out.add(
                    f"  %metadata_origin = index.constant {rows * width} : offset\n  %metadata = buffer.view %packed[%metadata_origin] : buffer -> view<{rows * width // 16}xf32>"
                )
                for row in range(
                    rows if a == c else max(0, a) * mult if 0 <= a <= c else 0
                ):
                    xs = xrow(row, width)
                    xs[0:4] = [127 / 32, 0.5 / 32, -0.5 / 32, -1.5 / 32]
                    qs, meta = quantize(xs)
                    assert qs[:4] == [127, 1, -1, -2]
                    for i, q in enumerate(qs):
                        if q:
                            out.put("packed", row * width + i, q % 256, total, "i8")
                    for g, (scale, raw_sum) in enumerate(meta):
                        if scale or raw_sum:
                            out.store(
                                "metadata",
                                (row * (width // 32) + g) * 2,
                                scale,
                                shape=f"{rows * width // 16}xf32",
                            )
                            out.store(
                                "metadata",
                                (row * (width // 32) + g) * 2 + 1,
                                raw_sum,
                                shape=f"{rows * width // 16}xf32",
                            )
                out.end()
                # Guard representation i32 checks exact packed bytes and scale bits.
                out.scenario(
                    name,
                    [
                        ("x", rows * width, "f32", "-777.0"),
                        ("count", 3, "i32", 99),
                        ("packed", total // 4, "i32", 999),
                    ],
                    ["packed"],
                )


def emit_combine(out):
    out.add(
        "kernel.decl @llm.routing.weighted_top8(%C: index) launch(%C0: index, %count: buffer, %ys: buffer, %weights: buffer, %out: buffer)"
    )
    for c in CAPACITIES:
        for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
            name = f'combine_c{c}_a{a if a >= 0 else "negative"}'
            args = ["ys", "weights", "count", "out"]
            out.func(name + ".init", args)
            count_init(out, a)
            out.view("ys", c * 8 * 2048)
            out.view("weights", c * 8)
            for t in range(c):
                for k in range(8):
                    wt = F((k + 1) / 36)
                    out.put("weights", t * 8 + k, wt, c * 8)
                    for ch in (0, 1, 2047):
                        y = F(
                            ([16, 1 / 17, -16, -1 / 31, 0.5, -0.25, 0.125, -0.0625][k])
                            * (1 + ch % 3)
                            * (1 + t % 2)
                        )
                        out.put("ys", (t * 8 + k) * 2048 + ch, y, c * 8 * 2048)
            out.end()
            out.command(
                name + ".actual",
                args,
                f"""  %C = index.constant {c} : index
  kernel.launch @llm.routing.weighted_top8[%C](%C, %count, %ys, %weights, %out) : [index](index, buffer, buffer, buffer, buffer)""",
            )
            out.func(name + ".expected", args)
            out.zero("out", c * 2048)
            for t in range(a if 0 <= a <= c else 0):
                for ch in (0, 1, 2047):
                    expected = 0.0
                    for k in range(8):
                        y = F(
                            ([16, 1 / 17, -16, -1 / 31, 0.5, -0.25, 0.125, -0.0625][k])
                            * (1 + ch % 3)
                            * (1 + t % 2)
                        )
                        expected = fmadd(F((k + 1) / 36), y, expected)
                    ideal = math.fsum(
                        F((k + 1) / 36)
                        * F(
                            ([16, 1 / 17, -16, -1 / 31, 0.5, -0.25, 0.125, -0.0625][k])
                            * (1 + ch % 3)
                            * (1 + t % 2)
                        )
                        for k in range(8)
                    )
                    assert abs(expected - ideal) <= 1e-4 + 5e-4 * abs(ideal)
                    out.put("out", t * 2048 + ch, expected, c * 2048)
            out.end()
            out.scenario(
                name,
                [
                    ("ys", c * 8 * 2048, "f32", "0.0"),
                    ("weights", c * 8, "f32", "0.0"),
                    ("count", 3, "i32", 99),
                    ("out", c * 2048, "f32", "77.0"),
                ],
                ["out"],
            )


def emit_activation(out):
    name = "activation"
    args = ["gate", "up", "out"]
    out.add(
        "kernel.decl @routed.activation() launch(%gate: buffer, %up: buffer, %out: buffer)"
    )
    out.func(name + ".init", args)
    for key in ("gate", "up"):
        out.view(key, 14)
    vals = [(g, u) for g in (-8, -4, -1, 0, 1, 4, 8) for u in (-0.75, 1.25)]
    for i, (g, u) in enumerate(vals):
        out.put("gate", i, g, 14)
        out.put("up", i, u, 14)
    out.end()
    out.command(
        name + ".actual",
        args,
        "  kernel.launch @routed.activation[](%gate, %up, %out) : [](buffer, buffer, buffer)",
    )
    out.func(name + ".expected", args)
    out.view("out", 14)
    for i, (g, u) in enumerate(vals):
        out.put("out", i, F(g / (1 + math.exp(-g)) * u), 14)
    out.end()
    out.scenario(
        name,
        [
            ("gate", 14, "f32", "0.0"),
            ("up", 14, "f32", "0.0"),
            ("out", 14, "f32", "77.0"),
        ],
        ["out"],
        approx={"out": (5e-4, 2e-3)},
    )


def emit_weight_initializer(
    out, name, tag, width, columns, up=False, middle=False, amplitude=1
):
    block_values, block_bytes = BLOCKS[tag]
    row_bytes = width // block_values * block_bytes
    expert_bytes = columns * row_bytes
    out.func(name, ["weight"])
    out.add(
        f"""  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %four = index.constant 4 : index
  %five = index.constant 5 : index
  %eight = index.constant 8 : index
  %sixteen = index.constant {18 if middle else 16} : index
  %last_column = index.constant {columns - 1} : index
  %blocks = index.constant {width // block_values} : index
  %last_expert_delta = index.constant 112 : index
  %row_stride = index.constant {row_bytes} : offset
  %expert_stride = index.constant {expert_bytes} : offset
  %block_stride = index.constant {block_bytes} : offset
  scf.for %e0 = [%zero to %sixteen step %one] {{
    %low = index.cmp ult, %e0, %eight : index
    %high = index.add %e0, %last_expert_delta : index
    %e_base = scf.select %low, %e0, %high : index
{'''    %ten = index.constant 10 : index
    %is_middle = index.cmp ult, %e0, %ten : index
    %middle_delta = index.constant 55 : index
    %middle = index.add %e0, %middle_delta : index
    %upper_delta = index.constant 110 : index
    %upper = index.add %e0, %upper_delta : index
    %not_low = scf.select %is_middle, %middle, %upper : index
    %e = scf.select %low, %e0, %not_low : index''' if middle else '    %e = index.add %e_base, %zero : index'}
    %eo = index.scale %e, %expert_stride : index, offset -> offset
    %parity = index.rem %e, %two : index
    %odd = index.cmp eq, %parity, %one : index
    scf.for %c = [%zero to %three step %one] {{
      %last = index.cmp eq, %c, %two : index
      %ch = scf.select %last, %last_column, %c : index
      %co = index.scale %ch, %row_stride : index, offset -> offset
      %row_origin = index.add %eo, %co : offset
      scf.for %b = [%zero to %blocks step %one] {{
        %bo = index.scale %b, %block_stride : index, offset -> offset
        %block_origin = index.add %row_origin, %bo : offset
        %bytes = buffer.view %weight[%block_origin] : buffer -> view<{block_bytes}xi8>"""
    )
    if tag == 12:
        d = 1 / 32 if width == 768 else 1 / 16 if up else 1 / 8
        out.add(
            f"""        %header = buffer.view %weight[%block_origin] : buffer -> view<2xf16>
        %d_even = scalar.constant {d} : f16
        %d_odd = scalar.constant {d * 1.5} : f16
        %m_even = scalar.constant {d / 4} : f16
        %m_odd = scalar.constant {d * 1.5 / 4} : f16
        %d = scf.select %odd, %d_odd, %d_even : f16
        %m = scf.select %odd, %m_odd, %m_even : f16
        view.store %d, %header[0] : f16, view<2xf16>
        view.store %m, %header[1] : f16, view<2xf16>
        %base = index.add %e, %ch : index
        %plus_block = index.add %base, %b : index
        %up_delta = index.constant {2 if up else 0} : index
        %selector = index.add %plus_block, %up_delta : index
        %modulo = index.rem %selector, %five : index
        %nibble = index.add %modulo, %five : index
        %ni = index.cast %nibble : index to i32
        %seventeen = scalar.constant 17 : i32
        %code32 = scalar.muli %ni, %seventeen : i32
        %code8 = scalar.trunci %code32 : i32 to i8
        %payload_start = index.constant 16 : index
        %end = index.constant 144 : index
        scf.for %i = [%payload_start to %end step %one] {{
          view.store %code8, %bytes[%i] : i8, view<144xi8>
        }}
        %scale_start = index.constant 4 : index
        %scale_end = index.constant 12 : index
        %s1 = scalar.constant 1 : i8
        scf.for %i = [%scale_start to %scale_end step %one] {{
          view.store %s1, %bytes[%i] : i8, view<144xi8>
        }}
        %s17 = scalar.constant 17 : i8
        scf.for %i = [%scale_end to %payload_start step %one] {{
          view.store %s17, %bytes[%i] : i8, view<144xi8>
        }}"""
        )
    elif tag == 14:
        out.add(
            """        %lo_end = index.constant 128 : index
        %hi_end = index.constant 192 : index
        %scale_end = index.constant 208 : index
        %ch_parity = index.rem %ch, %two : index
        %ch32 = index.cast %ch_parity : index to i32
        %base_lo = scalar.constant 166 : i32
        %lo32 = scalar.addi %base_lo, %ch32 : i32
        %lo8 = scalar.trunci %lo32 : i32 to i8
        %hi8 = scalar.constant 89 : i8
        scf.for %i = [%zero to %lo_end step %one] {
          view.store %lo8, %bytes[%i] : i8, view<210xi8>
        }
        scf.for %i = [%lo_end to %hi_end step %one] {
          view.store %hi8, %bytes[%i] : i8, view<210xi8>
        }
        %p1 = scalar.constant 1 : i8
        %p2 = scalar.constant 2 : i8
        %n1 = scalar.constant -1 : i8
        %n2 = scalar.constant -2 : i8
        %positive = scf.select %odd, %p2, %p1 : i8
        %negative = scf.select %odd, %n2, %n1 : i8
        scf.for %i = [%hi_end to %scale_end step %one] {
          %par = index.rem %i, %two : index
          %is_odd = index.cmp eq, %par, %one : index
          %scale = scf.select %is_odd, %negative, %positive : i8
          view.store %scale, %bytes[%i] : i8, view<210xi8>
        }
        %d_add = index.constant 208 : offset
        %d_origin = index.add %block_origin, %d_add : offset
        %d_view = buffer.view %weight[%d_origin] : buffer -> view<1xf16>
        %d = scalar.constant AMPLITUDE : f16
        view.store %d, %d_view[0] : f16, view<1xf16>""".replace(
                "AMPLITUDE", str(amplitude / 32)
            )
        )
    elif tag == 13:
        d = 1 / 64 if width == 768 or up else 1 / 32
        out.add(
            f"""        %header = buffer.view %weight[%block_origin] : buffer -> view<2xf16>
        %d_even = scalar.constant {d} : f16
        %d_odd = scalar.constant {d * 1.5} : f16
        %m_even = scalar.constant {d / 4} : f16
        %m_odd = scalar.constant {d * 1.5 / 4} : f16
        %d = scf.select %odd, %d_odd, %d_even : f16
        %m = scf.select %odd, %m_odd, %m_even : f16
        view.store %d, %header[0] : f16, view<2xf16>
        view.store %m, %header[1] : f16, view<2xf16>
        %base = index.add %e, %ch : index
        %plus_block = index.add %base, %b : index
        %up_delta = index.constant {2 if up else 0} : index
        %selector = index.add %plus_block, %up_delta : index
        %modulo = index.rem %selector, %five : index
        %nibble = index.add %modulo, %five : index
        %ni = index.cast %nibble : index to i32
        %seventeen = scalar.constant 17 : i32
        %code32 = scalar.muli %ni, %seventeen : i32
        %code8 = scalar.trunci %code32 : i32 to i8
        %scale_start = index.constant 4 : index
        %scale_mid = index.constant 12 : index
        %high_start = index.constant 16 : index
        %payload_start = index.constant 48 : index
        %end = index.constant 176 : index
        %s1 = scalar.constant 1 : i8
        %s17 = scalar.constant 17 : i8
        %s85 = scalar.constant 85 : i8
        scf.for %i = [%scale_start to %scale_mid step %one] {{
          view.store %s1, %bytes[%i] : i8, view<176xi8>
        }}
        scf.for %i = [%scale_mid to %high_start step %one] {{
          view.store %s17, %bytes[%i] : i8, view<176xi8>
        }}
        scf.for %i = [%high_start to %payload_start step %one] {{
          view.store %s85, %bytes[%i] : i8, view<176xi8>
        }}
        scf.for %i = [%payload_start to %end step %one] {{
          view.store %code8, %bytes[%i] : i8, view<176xi8>
        }}"""
        )
    elif tag in (8, 6):
        d = 1 / 64 if width == 768 or up else 1 / 32
        out.add(
            f"""        %header = buffer.view %weight[%block_origin] : buffer -> view<1xf16>
        %d_even = scalar.constant {d} : f16
        %d_odd = scalar.constant {d * 1.5} : f16
        %d = scf.select %odd, %d_odd, %d_even : f16
        view.store %d, %header[0] : f16, view<1xf16>
        %base = index.add %e, %ch : index
        %plus_block = index.add %base, %b : index
        %up_delta = index.constant {2 if up else 0} : index
        %selector = index.add %plus_block, %up_delta : index
        %modulo = index.rem %selector, %five : index
        %code = index.add %modulo, %five : index
        %code32 = index.cast %code : index to i32
        %zero32 = scalar.constant 0 : i32
        %neg32 = scalar.subi %zero32, %code32 : i32
        %one32 = scalar.constant 1 : i32
        %last32 = scalar.subi %code32, %one32 : i32"""
        )
        if tag == 8:
            out.add("""        %payload_start = index.constant 2 : index
        %end = index.constant 32 : index
        %format_last_lane = index.constant 31 : index
        scf.for %i = [%zero to %end step %one] {
          %negative = index.cmp eq, %i, %one : index
          %tail = index.cmp eq, %i, %format_last_lane : index
          %base_code = scf.select %negative, %neg32, %code32 : i32
          %value32 = scf.select %tail, %last32, %base_code : i32
          %value = scalar.trunci %value32 : i32 to i8
          %at = index.add %i, %payload_start : index
          view.store %value, %bytes[%at] : i8, view<34xi8>
        }""")
        else:
            out.add("""        %mask_start = index.constant 2 : index
        %mask = buffer.view %weight[%block_origin] : buffer -> view<22xi8>
        %byte_fd = scalar.constant -3 : i8
        %byte_ff = scalar.constant -1 : i8
        view.store %byte_fd, %mask[2] : i8, view<22xi8>
        view.store %byte_ff, %mask[3] : i8, view<22xi8>
        view.store %byte_ff, %mask[4] : i8, view<22xi8>
        view.store %byte_ff, %mask[5] : i8, view<22xi8>
        %six32 = scalar.constant 16 : i32
        %negative_nibble = scalar.subi %six32, %code32 : i32
        %shift = scalar.constant 4 : i32
        %end = index.constant 16 : index
        %format_last_lane = index.constant 15 : index
        %payload_start = index.constant 6 : index
        scf.for %i = [%zero to %end step %one] {
          %negative = index.cmp eq, %i, %one : index
          %tail = index.cmp eq, %i, %format_last_lane : index
          %low = scf.select %negative, %negative_nibble, %code32 : i32
          %high = scf.select %tail, %last32, %code32 : i32
          %shifted = scalar.shli %high, %shift : i32
          %value32 = scalar.ori %low, %shifted : i32
          %value = scalar.trunci %value32 : i32 to i8
          %at = index.add %i, %payload_start : index
          view.store %value, %bytes[%at] : i8, view<22xi8>
        }""")
    else:
        raise ValueError(tag)
    out.add("      }\n    }\n  }")
    out.end()


def put_packed(out, buffer, values, rows, width):
    total = rows * width * 5 // 4
    out.view(buffer, total, "i8")
    out.add(
        f"  %{buffer}_metadata_origin = index.constant {rows * width} : offset\n  %{buffer}_metadata = buffer.view %{buffer}[%{buffer}_metadata_origin] : buffer -> view<{rows * width // 16}xf32>"
    )
    for row, xs in enumerate(values):
        qs, metadata = quantize(xs)
        for i, q in enumerate(qs):
            if q:
                out.put(buffer, row * width + i, q % 256, total, "i8")
        for g, (scale, raw_sum) in enumerate(metadata):
            if scale or raw_sum:
                out.store(
                    buffer + "_metadata",
                    2 * (row * width // 32 + g),
                    scale,
                    shape=f"{rows * width // 16}xf32",
                )
                out.store(
                    buffer + "_metadata",
                    2 * (row * width // 32 + g) + 1,
                    raw_sum,
                    shape=f"{rows * width // 16}xf32",
                )


def expert_hidden(
    row, expert, xs=None, gate_tag=12, up_tag=12, gate_amplitude=1, up_amplitude=1
):
    xs = xrow(row) if xs is None else xs
    dots = [
        (
            ch,
            dot(gate_tag, expert, ch, xs, amplitude=gate_amplitude),
            dot(up_tag, expert, ch, xs, True, amplitude=up_amplitude),
        )
        for ch in (0, 1, 767)
    ]
    values = [0.0] * 768
    for ch, gate, up in dots:
        assert abs(gate) <= 8 and abs(up) <= 8, (gate, up)
        values[ch] = F(gate / (1 + math.exp(-gate)) * up)
    quantize(values)  # Composite fixture must itself remain in frozen pack domain.
    return dots, values


def emit_indexed(out):
    emit_weight_initializer(out, "weight_gate", 12, 2048, 768)
    emit_weight_initializer(out, "weight_up", 12, 2048, 768, True)
    emit_weight_initializer(out, "weight_down4", 12, 768, 2048)
    emit_weight_initializer(out, "weight_down6", 14, 768, 2048)
    out.add(
        "kernel.decl @routed.observe_gate_up(%N: index, %R: index) launch(%N0: index, %R0: index, %count: buffer, %ids: buffer, %packed: buffer, %gate: buffer, %up: buffer, %hidden: buffer, %gate_out: buffer, %up_out: buffer)"
    )
    for tag in (12, 14):
        out.add(
            f"kernel.decl @llm.experts.indexed_down_q{4 if tag == 12 else 6}(%N: index, %R: index) launch(%N0: index, %R0: index, %count: buffer, %ids: buffer, %packed: buffer, %down: buffer, %out: buffer)"
        )
    for c in CAPACITIES:
        for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
            rows = 8 * c
            valid_rows = a * 8 if 0 <= a <= c else 0
            ids = [route(t, 1 if c in (8, 17) else 0)[2] for t in range(c)]
            if c == 1 and a == 1:
                # Leaf-only clamp defense and duplicates; public composition
                # still accepts only the IDs its router produces.
                ids = [[-3, 128, 127, 0, 127, 0, 120, 121]]
            for kind in ("gate", "down4", "down6"):
                name = f'indexed_{kind}_c{c}_a{a if a >= 0 else "negative"}'
                gate = kind == "gate"
                tag = 14 if kind == "down6" else 12
                width = 2048 if gate else 768
                columns = 768 if gate else 2048
                prows = c if gate else rows
                packed_bytes = prows * width * 5 // 4
                wb = 113246208 if tag == 12 else 165150720
                outputs = ["hidden", "gate_out", "up_out"] if gate else ["out"]
                args = (
                    ["packed", "ids", "count", "gate", "up"] + outputs
                    if gate
                    else ["packed", "ids", "count", "down", "out"]
                )
                out.func(name + ".init", args)
                count_init(out, a)
                out.view("ids", rows, "i32")
                for t in range(c):
                    for k, eid in enumerate(ids[t]):
                        out.put("ids", t * 8 + k, eid, rows, "i32")
                exact_dots = c == 1 and a == 1
                fixed_values = [xrow(t, width) for t in range(prows)]
                if exact_dots:
                    for xs in fixed_values:
                        xs[0], xs[1], xs[2], xs[-1] = 127 / 128, -0.5, 0.25, 127 / 128
                put_packed(out, "packed", fixed_values, prows, width)
                if gate:
                    out.add(
                        "  func.call @weight_gate(%gate) : (buffer)\n  func.call @weight_up(%up) : (buffer)"
                    )
                else:
                    out.add(f"  func.call @weight_{kind}(%down) : (buffer)")
                out.end()
                body = f"  %R = index.constant {rows} : index\n  %N = index.constant {columns} : index\n"
                if gate:
                    body += "  kernel.launch @routed.observe_gate_up[%N, %R](%N, %R, %count, %ids, %packed, %gate, %up, %hidden, %gate_out, %up_out) : [index, index](index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer)"
                else:
                    body += f"  kernel.launch @llm.experts.indexed_down_q{4 if tag == 12 else 6}[%N, %R](%N, %R, %count, %ids, %packed, %down, %out) : [index, index](index, index, buffer, buffer, buffer, buffer, buffer)"
                out.command(name + ".actual", args, body)
                out.func(name + ".expected", args)
                for key in outputs:
                    out.zero(key, rows * columns)
                for row in range(valid_rows):
                    expert = max(0, min(127, ids[row // 8][row % 8]))
                    if gate:
                        dots, hidden = expert_hidden(
                            row // 8, expert, fixed_values[row // 8]
                        )
                        for ch, gd, ud in dots:
                            out.put("gate_out", row * 768 + ch, gd, rows * 768)
                            out.put("up_out", row * 768 + ch, ud, rows * 768)
                            out.put("hidden", row * 768 + ch, hidden[ch], rows * 768)
                    else:
                        for ch in (0, 1, 2047):
                            value = dot(tag, expert, ch, fixed_values[row])
                            assert abs(value) <= 16, value
                            out.put("out", row * 2048 + ch, value, rows * 2048)
                out.end()
                tensors = [
                    ("packed", packed_bytes // 4, "i32", 0),
                    ("ids", rows, "i32", -999),
                    ("count", 3, "i32", 99),
                ]
                tensors += (
                    [("gate", wb, "i8", 0), ("up", wb, "i8", 0)]
                    if gate
                    else [("down", wb, "i8", 0)]
                )
                tensors += [(key, rows * columns, "f32", "99.0") for key in outputs]
                budgets = {
                    key: ((5e-4, 2e-3) if key == "hidden" else (1e-4, 5e-4))
                    for key in outputs
                }
                if exact_dots:
                    budgets = {"hidden": (5e-4, 2e-3)} if gate else {}
                out.scenario(name, tensors, outputs, approx=budgets)


def emit_composition(out):
    out.add(
        "template.decl @llm.ffn.direct_routed(%C: index, %GT: index, %UT: index, %tag: index, %GB: index, %UB: index, %DB: index, %count: buffer, %input: buffer, %router: view<128x2048xf32>, %gate: view<[%GB]xi8>, %up: view<[%UB]xi8>, %down: view<[%DB]xi8>, %out: buffer)"
    )
    for c in CAPACITIES:
        for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
            for tag in (12, 14):
                name = f'composition_q{tag}_c{c}_a{a if a >= 0 else "negative"}'
                db = 113246208 if tag == 12 else 165150720
                args = ["x", "router", "gate", "up", "down", "count", "out"]
                out.func(name + ".init", args)
                count_init(out, a)
                out.zero("x", c * 2048)
                out.view("router", 128 * 2048)
                for t in range(c):
                    for ch, v in enumerate(xrow(t)):
                        if v:
                            out.put(
                                "x",
                                t * 2048 + ch,
                                v if 0 <= t < a <= c else -777.0,
                                c * 2048,
                            )
                for e in range(128):
                    w = 1 + 0.25 * (e - 120) if e >= 120 else -3 + e / 128
                    out.put("router", e * 2048, w, 128 * 2048)
                out.add(
                    f"  func.call @weight_gate(%gate) : (buffer)\n  func.call @weight_up(%up) : (buffer)\n  func.call @weight_down{4 if tag == 12 else 6}(%down) : (buffer)"
                )
                out.end()
                body = f"""  %C = index.constant {c} : index
  %tag = index.constant {tag} : index
  %q4 = index.constant 12 : index
  %GB = index.constant 113246208 : index
  %DB = index.constant {db} : index
  %origin = index.constant 0 : offset
  %rv = buffer.view %router[%origin] : buffer -> view<128x2048xf32>
  %gv = buffer.view %gate[%origin] : buffer -> view<[%GB]xi8>
  %uv = buffer.view %up[%origin] : buffer -> view<[%GB]xi8>
  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>
  template.apply<@llm.ffn.direct_routed>(%C, %q4, %q4, %tag, %GB, %GB, %DB, %count, %x, %rv, %gv, %uv, %dv, %out) : (index, index, index, index, index, index, index, buffer, buffer, view<128x2048xf32>, view<[%GB]xi8>, view<[%GB]xi8>, view<[%DB]xi8>, buffer)"""
                out.command(name + ".actual", args, body)
                out.func(name + ".expected", args)
                out.zero("out", c * 2048)
                for t in range(a if 0 <= a <= c else 0):
                    _, _, ids, weights = route(t, 1)
                    hidden = [expert_hidden(t, e)[1] for e in ids]
                    for ch in (0, 1, 2047):
                        result = 0.0
                        for k, e in enumerate(ids):
                            y = dot(tag, e, ch, hidden[k])
                            assert abs(y) <= 16, y
                            result = fmadd(weights[k], y, result)
                        out.put("out", t * 2048 + ch, result, c * 2048)
                out.end()
                tensors = [
                    ("x", c * 2048, "f32", "0.0"),
                    ("router", 128 * 2048, "f32", "0.0"),
                    ("gate", 113246208, "i8", 0),
                    ("up", 113246208, "i8", 0),
                    ("down", db, "i8", 0),
                    ("count", 3, "i32", 99),
                    ("out", c * 2048, "f32", "99.0"),
                ]
                out.scenario(name, tensors, ["out"], approx={"out": (1e-3, 3e-3)})


def emit_raw_codes(out):
    out.add(
        "kernel.decl @routed.raw_codes(%tag: index) launch(%tag0: index, %weight: buffer, %codes: buffer, %metadata: buffer)"
    )
    for tag in (12, 14):
        size = {12: 144, 14: 210}[tag]
        data = bytearray((i * 73 + 29) % 256 for i in range(size))
        if tag == 12:
            struct.pack_into("<ee", data, 0, 0.125, -0.0625)
        else:
            struct.pack_into("<e", data, 208, -0.125)
        name = f"wire_q{tag}"
        args = ["weight", "codes", "metadata"]
        out.func(name + ".init", args)
        out.view("weight", size, "i8")
        for i, b in enumerate(data):
            out.put("weight", i, b, size, "i8")
        out.end()
        out.command(
            name + ".actual",
            args,
            f"  %tag = index.constant {tag} : index\n  kernel.launch @routed.raw_codes[%tag](%tag, %weight, %codes, %metadata) : [index](index, buffer, buffer, buffer)",
        )
        out.func(name + ".expected", args)
        out.view("codes", 256)
        out.view("metadata", 16)
        for k in range(256):
            code, scale, minimum = decode(data, tag, k)
            out.put("codes", k, code, 256)
        for g in range(8):
            _, scale, minimum = decode(data, tag, g * 32)
            if tag == 14:
                _, minimum, _ = decode(data, tag, g * 32 + 16)
            out.put("metadata", g * 2, scale, 16)
            out.put("metadata", g * 2 + 1, minimum, 16)
        out.end()
        out.scenario(
            name,
            [
                ("weight", size, "i8", 0),
                ("codes", 256, "f32", "999.0"),
                ("metadata", 16, "f32", "99.0"),
            ],
            ["codes", "metadata"],
        )


def emit_extreme_router(out):
    # Separate finite-logit safety regression, outside the bounded accuracy domain.
    # Sparse F32 dot is exactly -15*2^124 and cannot be dominated by padded lanes.
    name = "route_extreme_finite_tie"
    value = F(-15 * 2**124)
    args = ["x", "router", "count", "ids", "weights", "logits", "probabilities"]
    out.add(
        "kernel.decl @routed.observe_route(%C: index) launch(%C0: index, %x: buffer, %w: buffer, %count: buffer, %ids: buffer, %weights: buffer, %logits: buffer, %probabilities: buffer)"
    )
    out.func(name + ".init", args)
    count_init(out, 1)
    out.view("x", 2048)
    out.put("x", 0, 1.0, 2048)
    out.view("router", 128 * 2048)
    for e in range(128):
        out.put("router", e * 2048, value, 128 * 2048)
    out.end()
    out.command(
        name + ".actual",
        args,
        "  %C = index.constant 1 : index\n  kernel.launch @routed.observe_route[%C](%C, %x, %router, %count, %ids, %weights, %logits, %probabilities) : [index](index, buffer, buffer, buffer, buffer, buffer, buffer, buffer)",
    )
    out.func(name + ".expected", args)
    for key, size, typ in [
        ("ids", 8, "i32"),
        ("weights", 8, "f32"),
        ("logits", 128, "f32"),
        ("probabilities", 128, "f32"),
    ]:
        out.view(key, size, typ)
    for k in range(8):
        out.put("ids", k, k, 8, "i32")
        out.put("weights", k, 0.125, 8)
    for e in range(128):
        out.put("logits", e, value, 128)
        out.put("probabilities", e, 1 / 128, 128)
    out.end()
    out.scenario(
        name,
        [
            ("x", 2048, "f32", "0.0"),
            ("router", 128 * 2048, "f32", "0.0"),
            ("count", 3, "i32", 99),
            ("ids", 8, "i32", 999),
            ("weights", 8, "f32", "99.0"),
            ("logits", 128, "f32", "99.0"),
            ("probabilities", 128, "f32", "99.0"),
        ],
        ["ids", "weights", "logits", "probabilities"],
    )


def emit_denominator(out):
    name = "denominator_clamp"
    args = ["probabilities", "out"]
    out.add(
        "kernel.decl @routed.denominator() launch(%probabilities: buffer, %out: buffer)"
    )
    values = [
        [0.0] * 8,
        [F((k + 1) * 2**-24) for k in range(8)],
        [F((k + 1) / 31) for k in range(8)],
    ]
    out.func(name + ".init", args)
    out.view("probabilities", 24)
    for row, xs in enumerate(values):
        for k, v in enumerate(xs):
            out.put("probabilities", row * 8 + k, v, 24)
    out.end()
    out.command(
        name + ".actual",
        args,
        "  kernel.launch @routed.denominator[](%probabilities, %out) : [](buffer, buffer)",
    )
    out.func(name + ".expected", args)
    out.view("out", 24)
    for row, xs in enumerate(values):
        total = 0.0
        for v in xs:
            total = F(total + v)
        for k, v in enumerate(xs):
            out.put("out", row * 8 + k, F(v / max(total, 2**-14)), 24)
    out.end()
    out.scenario(
        name, [("probabilities", 24, "f32", "0.0"), ("out", 24, "f32", "77.0")], ["out"]
    )


def single_symbol(tag, gate=False):
    suffix = {12: "q4", 13: "q5", 14: "q6", 8: "q8", 6: "q5_0"}[tag]
    return "llm.experts.indexed_" + ("gate_" if gate else "down_") + suffix


def paired_symbol(gate, up, observed=False):
    prefix = "routed.observe_gate_up" if observed else "llm.experts.indexed_swiglu"
    if gate == up == 12:
        return prefix if observed else prefix + "_q4"
    return prefix + f"_g{gate}_u{up}"


def leaf_declaration(out, symbol, paired=False, observed=False):
    arguments = (
        "%N0:index,%R0:index,%count:buffer,%ids:buffer,%packed:buffer,%w:buffer,"
    )
    if paired:
        arguments += "%u:buffer,"
    arguments += "%out:buffer"
    if observed:
        arguments += ",%gate_out:buffer,%up_out:buffer"
    out.add(f"kernel.decl @{symbol}(%N:index,%R:index) launch({arguments})")


def leaf_launch(symbol, weights, output, observed=()):
    names = ["%N", "%R", "%count", "%ids", "%packed"]
    names += ["%" + key for key in weights + [output] + list(observed)]
    return (
        f"  kernel.launch @{symbol}[%N,%R]("
        + ",".join(names)
        + ") : [index,index](index,index,"
        + ",".join(["buffer"] * (len(names) - 2))
        + ")"
    )


def compare_rows(out, left, right, flags, exact=False, atol=5e-4, rtol=2e-3):
    out.add(
        f"""  %exact_{flags} = scalar.constant {int(exact)} : i32
  %atol_{flags} = scalar.constant {atol} : f32
  %rtol_{flags} = scalar.constant {rtol} : f32
  kernel.launch @routed.compare_rows[%R,%N](%R,%N,%exact_{flags},%atol_{flags},%rtol_{flags},%{left},%{right},%{flags}) : [index,index](index,index,i32,f32,f32,buffer,buffer,buffer)"""
    )


def comparison_declarations(out):
    out.add(
        "kernel.decl @routed.compare_rows(%R:index,%N:index) launch(%R0:index,%N0:index,%exact:i32,%atol:f32,%rtol:f32,%left:buffer,%right:buffer,%flags:buffer)"
    )
    out.add(
        "kernel.decl @routed.swiglu_rows(%R:index,%N:index) launch(%R0:index,%N0:index,%gate:buffer,%up:buffer,%out:buffer)"
    )


def selected_ids(c):
    order = (0, 63, 127, 64, 127, 0, 63, 1)
    return [[order[(t + k) % 8] for k in range(8)] for t in range(c)]


def init_indexed_inputs(
    out, name, args, c, active, width, gate, initializers, exact=False
):
    rows = 8 * c
    prows = c if gate else rows
    values = [xrow(t, width) for t in range(prows)]
    if exact:
        for xs in values:
            xs[0], xs[1], xs[2], xs[-1] = 127 / 128, -0.5, 0.25, 127 / 128
    ids = selected_ids(c)
    out.func(name + ".init", args)
    count_init(out, active)
    out.view("ids", rows, "i32")
    for t in range(c):
        for k, expert in enumerate(ids[t]):
            out.put("ids", 8 * t + k, expert, rows, "i32")
    put_packed(out, "packed", values, prows, width)
    for initializer, buffer in initializers:
        out.add(f"  func.call @{initializer}(%{buffer}) : (buffer)")
    out.end()
    tensors = [
        ("packed", prows * width * 5 // 16, "i32", 0),
        ("ids", rows, "i32", -999),
        ("count", 3, "i32", 99),
    ]
    return ids, values, tensors


def emit_formats_wire(out):
    for tag, suffix in ((13, "q5"), (8, "q8"), (6, "q5_0")):
        out.add(
            f"kernel.decl @routed.format_codes_{suffix}() launch(%weight:buffer,%codes:buffer,%metadata:buffer)"
        )
        size = BLOCKS[tag][1]
        data = bytearray((73 * i + 29) % 256 for i in range(size))
        if tag == 13:
            struct.pack_into("<ee", data, 0, -0.125, 0.0625)
        else:
            struct.pack_into("<e", data, 0, -0.125)
        name = f"format_wire_q{tag}"
        args = ["weight", "codes", "metadata"]
        out.func(name + ".init", args)
        out.view("weight", size, "i8")
        for i, byte in enumerate(data):
            out.put("weight", i, byte, size, "i8")
        out.end()
        out.command(
            name + ".actual",
            args,
            f"  kernel.launch @routed.format_codes_{suffix}[](%weight,%codes,%metadata) : [](buffer,buffer,buffer)",
        )
        out.func(name + ".expected", args)
        out.view("codes", 256)
        out.view("metadata", 16)
        for k in range(BLOCKS[tag][0]):
            code, _, _ = decode(data, tag, k)
            out.put("codes", k, code, 256)
        for group in range(BLOCKS[tag][0] // 32):
            _, scale, minimum = decode(data, tag, group * 32)
            out.put("metadata", 2 * group, scale, 16)
            out.put("metadata", 2 * group + 1, minimum, 16)
        out.end()
        out.scenario(
            name,
            [
                ("weight", size, "i8", 0),
                ("codes", 256, "f32", "999.0"),
                ("metadata", 16, "f32", "99.0"),
            ],
            ["codes", "metadata"],
        )


def format_projection_amplitude(tag):
    # The signed Q6 witness has cancellation between its positive/negative
    # subgroups. A power-of-two header scale gives its new mixed-pair corpus
    # comparable preactivation magnitude to the unsigned K-format witnesses,
    # keeping nonzero hidden pack groups inside the unchanged finite domain.
    # Existing E1 and old/new parity calls retain their default amplitude one.
    return 16 if tag == 14 else 1


def emit_formats_pair(out, gate_tag, up_tag, c=1):
    assert gate_tag in GATE_TAGS and up_tag in GATE_TAGS
    ga, ua = map(format_projection_amplitude, (gate_tag, up_tag))
    emit_weight_initializer(
        out, "format_gate", gate_tag, 2048, 768, middle=True, amplitude=ga
    )
    emit_weight_initializer(
        out, "format_up", up_tag, 2048, 768, up=True, middle=True, amplitude=ua
    )
    pair = paired_symbol(gate_tag, up_tag)
    observer = paired_symbol(gate_tag, up_tag, observed=True)
    single_gate, single_up = single_symbol(gate_tag, True), single_symbol(up_tag, True)
    leaf_declaration(out, pair, paired=True)
    leaf_declaration(out, observer, paired=True, observed=True)
    for symbol in sorted(set((single_gate, single_up))):
        leaf_declaration(out, symbol)
    comparison_declarations(out)
    rows, columns = 8 * c, 768
    float_outputs = [
        "hidden",
        "observed_hidden",
        "gate_out",
        "up_out",
        "gate_single",
        "up_single",
        "separate_hidden",
    ]
    flag_outputs = ["pair_flags", "gate_flags", "up_flags"]
    outputs = float_outputs + flag_outputs
    args = ["packed", "ids", "count", "gate", "up"] + outputs
    for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
        name = f'format_pair_g{gate_tag}_u{up_tag}_c{c}_a{a if a >= 0 else "negative"}'
        exact = c == a == 1
        ids, values, tensors = init_indexed_inputs(
            out,
            name,
            args,
            c,
            a,
            2048,
            True,
            [("format_gate", "gate"), ("format_up", "up")],
            exact,
        )
        body = f"  %N = index.constant {columns} : index\n  %R = index.constant {rows} : index\n"
        body += leaf_launch(pair, ["gate", "up"], "hidden") + "\n"
        body += (
            leaf_launch(
                observer, ["gate", "up"], "observed_hidden", ("gate_out", "up_out")
            )
            + "\n"
        )
        body += leaf_launch(single_gate, ["gate"], "gate_single") + "\n"
        body += leaf_launch(single_up, ["up"], "up_single") + "\n"
        body += "  kernel.launch @routed.swiglu_rows[%R,%N](%R,%N,%gate_single,%up_single,%separate_hidden) : [index,index](index,index,buffer,buffer,buffer)\n"
        # Comparisons execute on the two real operation outputs. Expected flags
        # remain independently fixed zero; scalar goldens also check each output.
        compare = Emit()
        compare.lines = []
        compare_rows(compare, "hidden", "separate_hidden", "pair_flags")
        compare_rows(
            compare, "gate_out", "gate_single", "gate_flags", exact, 1e-4, 5e-4
        )
        compare_rows(compare, "up_out", "up_single", "up_flags", exact, 1e-4, 5e-4)
        out.command(name + ".actual", args, body + "\n".join(compare.lines))
        out.func(name + ".expected", args)
        for key in float_outputs:
            out.zero(key, rows * columns)
        for key in flag_outputs:
            out.zero(key, rows * columns, "i32")
        for row in range(8 * a if 0 <= a <= c else 0):
            expert = ids[row // 8][row % 8]
            dots, hidden = expert_hidden(
                row // 8, expert, values[row // 8], gate_tag, up_tag, ga, ua
            )
            for ch, gd, ud in dots:
                for key, value in (
                    ("gate_out", gd),
                    ("gate_single", gd),
                    ("up_out", ud),
                    ("up_single", ud),
                    ("hidden", hidden[ch]),
                    ("observed_hidden", hidden[ch]),
                    ("separate_hidden", hidden[ch]),
                ):
                    out.put(key, row * columns + ch, value, rows * columns)
        out.end()
        tensors += [
            ("gate", weight_bytes(gate_tag), "i8", 0),
            ("up", weight_bytes(up_tag), "i8", 0),
        ]
        tensors += [(key, rows * columns, "f32", "99.0") for key in float_outputs]
        tensors += [(key, rows * columns, "i32", 77) for key in flag_outputs]
        approx = {
            key: ((5e-4, 2e-3) if "hidden" in key else (1e-4, 5e-4))
            for key in float_outputs
            if not exact or "hidden" in key
        }
        out.scenario(name, tensors, outputs, approx=approx)


def emit_formats_down(out, tag, c=1):
    assert tag in DOWN_TAGS
    emit_weight_initializer(out, "format_down", tag, 768, 2048, middle=True)
    symbol = single_symbol(tag)
    leaf_declaration(out, symbol)
    rows = 8 * c
    args = ["packed", "ids", "count", "down", "out"]
    for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
        name = f'format_down_q{tag}_c{c}_a{a if a >= 0 else "negative"}'
        exact = c == a == 1
        ids, values, tensors = init_indexed_inputs(
            out, name, args, c, a, 768, False, [("format_down", "down")], exact
        )
        out.command(
            name + ".actual",
            args,
            f"  %N = index.constant 2048 : index\n  %R = index.constant {rows} : index\n"
            + leaf_launch(symbol, ["down"], "out"),
        )
        out.func(name + ".expected", args)
        out.zero("out", rows * 2048)
        for row in range(8 * a if 0 <= a <= c else 0):
            expert = ids[row // 8][row % 8]
            for ch in (0, 1, 2047):
                value = dot(tag, expert, ch, values[row])
                assert abs(value) <= 16, value
                out.put("out", row * 2048 + ch, value, rows * 2048)
        out.end()
        tensors += [
            ("down", weight_bytes(tag), "i8", 0),
            ("out", rows * 2048, "f32", "99.0"),
        ]
        out.scenario(
            name, tensors, ["out"], approx={} if exact else {"out": (1e-4, 5e-4)}
        )


def emit_formats_composition(out, gate_tag, up_tag, down_tag, c=1):
    assert gate_tag in GATE_TAGS and up_tag in GATE_TAGS and down_tag in DOWN_TAGS
    ga, ua = map(format_projection_amplitude, (gate_tag, up_tag))
    emit_weight_initializer(out, "format_gate", gate_tag, 2048, 768, amplitude=ga)
    emit_weight_initializer(out, "format_up", up_tag, 2048, 768, up=True, amplitude=ua)
    emit_weight_initializer(out, "format_down", down_tag, 768, 2048)
    out.add(
        "template.decl @llm.ffn.direct_routed(%C:index,%GT:index,%UT:index,%DT:index,%GB:index,%UB:index,%DB:index,%count:buffer,%x:buffer,%router:view<128x2048xf32>,%gate:view<[%GB]xi8>,%up:view<[%UB]xi8>,%down:view<[%DB]xi8>,%out:buffer)"
    )
    gb, ub, db = [weight_bytes(tag) for tag in (gate_tag, up_tag, down_tag)]
    args = ["x", "router", "gate", "up", "down", "count", "out"]
    for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
        name = f'format_composition_{gate_tag}_{up_tag}_{down_tag}_c{c}_a{a if a >= 0 else "negative"}'
        out.func(name + ".init", args)
        count_init(out, a)
        out.zero("x", c * 2048)
        out.view("router", 128 * 2048)
        for t in range(c):
            for ch, v in enumerate(xrow(t)):
                if v:
                    out.put(
                        "x", t * 2048 + ch, v if 0 <= t < a <= c else -777.0, c * 2048
                    )
        for expert in range(128):
            value = 1 + 0.25 * (expert - 120) if expert >= 120 else -3 + expert / 128
            out.put("router", expert * 2048, value, 128 * 2048)
        out.add(
            "  func.call @format_gate(%gate) : (buffer)\n"
            "  func.call @format_up(%up) : (buffer)\n"
            "  func.call @format_down(%down) : (buffer)"
        )
        out.end()
        out.command(
            name + ".actual",
            args,
            f"""  %C = index.constant {c} : index
  %GT = index.constant {gate_tag} : index
  %UT = index.constant {up_tag} : index
  %DT = index.constant {down_tag} : index
  %GB = index.constant {gb} : index
  %UB = index.constant {ub} : index
  %DB = index.constant {db} : index
  %origin = index.constant 0 : offset
  %rv = buffer.view %router[%origin] : buffer -> view<128x2048xf32>
  %gv = buffer.view %gate[%origin] : buffer -> view<[%GB]xi8>
  %uv = buffer.view %up[%origin] : buffer -> view<[%UB]xi8>
  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>
  template.apply<@llm.ffn.direct_routed>(%C,%GT,%UT,%DT,%GB,%UB,%DB,%count,%x,%rv,%gv,%uv,%dv,%out) : (index,index,index,index,index,index,index,buffer,buffer,view<128x2048xf32>,view<[%GB]xi8>,view<[%UB]xi8>,view<[%DB]xi8>,buffer)""",
        )
        out.func(name + ".expected", args)
        out.zero("out", c * 2048)
        for t in range(a if 0 <= a <= c else 0):
            _, _, ids, weights = route(t, 1)
            hidden = [
                expert_hidden(
                    t,
                    expert,
                    gate_tag=gate_tag,
                    up_tag=up_tag,
                    gate_amplitude=ga,
                    up_amplitude=ua,
                )[1]
                for expert in ids
            ]
            for ch in (0, 1, 2047):
                result = 0.0
                for k, expert in enumerate(ids):
                    value = dot(down_tag, expert, ch, hidden[k])
                    assert abs(value) <= 16, value
                    result = fmadd(weights[k], value, result)
                out.put("out", t * 2048 + ch, result, c * 2048)
        out.end()
        out.scenario(
            name,
            [
                ("x", c * 2048, "f32", "0.0"),
                ("router", 128 * 2048, "f32", "0.0"),
                ("gate", gb, "i8", 0),
                ("up", ub, "i8", 0),
                ("down", db, "i8", 0),
                ("count", 3, "i32", 99),
                ("out", c * 2048, "f32", "99.0"),
            ],
            ["out"],
            approx={"out": (1e-3, 3e-3)},
        )


def emit_parity(out, baseline_path, expected_sha256, kind, c):
    """Temporary external old/new check; never saves observed GPU expectations.

    CENTRAL supplies the immutable accepted indexed.loom blob and its recorded
    SHA256. This mode embeds that old body only into a disposable test overlay;
    no copied old production kernels are checked into this repository.
    """
    baseline = Path(baseline_path).read_bytes()
    assert hashlib.sha256(baseline).hexdigest() == expected_sha256
    reference = baseline.decode().replace(
        "@llm.experts.indexed_", "@routed.reference.indexed_"
    )
    out.add("// Exact accepted reference indexed.loom SHA256 " + expected_sha256)
    out.add(reference)
    assert kind in ("gate", "down4", "down6") and c in (1, 17)
    gate = kind == "gate"
    tag = 14 if kind == "down6" else 12
    width, columns = (2048, 768) if gate else (768, 2048)
    emit_weight_initializer(out, "parity_weight", tag, width, columns, middle=True)
    if gate:
        emit_weight_initializer(
            out, "parity_up", 12, width, columns, up=True, middle=True
        )
    symbol = paired_symbol(12, 12) if gate else single_symbol(tag)
    leaf_declaration(out, symbol, paired=gate)
    comparison_declarations(out)
    rows = 8 * c
    weights = ["weight", "up"] if gate else ["weight"]
    args = ["packed", "ids", "count"] + weights + ["new", "old", "flags"]
    initializers = [("parity_weight", "weight")]
    if gate:
        initializers.append(("parity_up", "up"))
    for a in sorted(set((0, 1, c // 2, c, -1, c + 1))):
        name = f'parity_{kind}_c{c}_a{a if a >= 0 else "negative"}'
        exact = c == a == 1
        ids, values, tensors = init_indexed_inputs(
            out, name, args, c, a, width, gate, initializers, exact
        )
        command = Emit()
        command.lines = []
        command.add(
            f"  %N = index.constant {columns} : index\n  %R = index.constant {rows} : index"
        )
        command.add(leaf_launch(symbol, weights, "new"))
        old_symbol = symbol.replace("llm.experts.indexed_", "routed.reference.indexed_")
        command.add(leaf_launch(old_symbol, weights, "old"))
        compare_rows(command, "new", "old", "flags", exact=True, atol=0.0, rtol=0.0)
        out.command(name + ".actual", args, "".join(command.lines))
        out.func(name + ".expected", args)
        out.zero("new", rows * columns)
        out.zero("old", rows * columns)
        out.zero("flags", rows * columns, "i32")
        for row in range(8 * a if 0 <= a <= c else 0):
            expert = ids[row // 8][row % 8]
            if gate:
                dots, hidden = expert_hidden(row // 8, expert, values[row // 8])
                expected = [(ch, hidden[ch]) for ch, _, _ in dots]
            else:
                expected = [
                    (ch, dot(tag, expert, ch, values[row]))
                    for ch in (0, 1, columns - 1)
                ]
            for ch, value in expected:
                for key in ("new", "old"):
                    out.put(key, row * columns + ch, value, rows * columns)
        out.end()
        tensors += [
            (key, weight_bytes(tag if key == "weight" else 12), "i8", 0)
            for key in weights
        ]
        tensors += [
            ("new", rows * columns, "f32", "99.0"),
            ("old", rows * columns, "f32", "99.0"),
            ("flags", rows * columns, "i32", 77),
        ]
        budget = (5e-4, 2e-3) if gate else (1e-4, 5e-4)
        approx = {"new": budget, "old": budget} if gate or not exact else {}
        out.scenario(name, tensors, ["new", "old", "flags"], approx=approx)


def main():
    global CAPACITIES
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    formats = stage.startswith("formats_") or stage == "parity"
    CAPACITIES = (
        (1,)
        if formats
        else (int(sys.argv[3]),) if len(sys.argv) > 3 else (1, 8, 16, 17)
    )
    out = Emit()
    if stage in ("router", "all"):
        emit_router(out)
    if stage in ("pack", "all"):
        emit_pack(out)
    if stage in ("combine", "all"):
        emit_combine(out)
    if stage in ("activation", "all"):
        emit_activation(out)
        emit_denominator(out)
        emit_raw_codes(out)
        emit_extreme_router(out)
    if stage in ("indexed", "all"):
        emit_indexed(out)
    if stage in ("composition", "all"):
        if stage != "all":
            emit_weight_initializer(out, "weight_gate", 12, 2048, 768)
            emit_weight_initializer(out, "weight_up", 12, 2048, 768, True)
            emit_weight_initializer(out, "weight_down4", 12, 768, 2048)
            emit_weight_initializer(out, "weight_down6", 14, 768, 2048)
        emit_composition(out)
    if stage == "formats_wire":
        emit_formats_wire(out)
    if stage == "formats_pair":
        emit_formats_pair(
            out,
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 1,
        )
    if stage == "formats_down":
        emit_formats_down(
            out, int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1
        )
    if stage == "formats_composition":
        emit_formats_composition(
            out,
            *(int(x) for x in sys.argv[3:6]),
            int(sys.argv[6]) if len(sys.argv) > 6 else 1,
        )
    if stage == "parity":
        emit_parity(out, sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
    assert stage in (
        "all",
        "router",
        "pack",
        "combine",
        "activation",
        "indexed",
        "composition",
        "formats_wire",
        "formats_pair",
        "formats_down",
        "formats_composition",
        "parity",
    ), stage
    Path(sys.argv[1]).write_text("".join(out.lines))


if __name__ == "__main__":
    main()
