#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent head RMS/NeoX and F64 causal GQA anchors at actual geometry."""

import math
from pathlib import Path
import struct
import sys
from reference import f32, half


def normalized(values, gamma):
    inverse = 1.0 / math.sqrt(
        sum(float(v) ** 2 for v in values) / len(values) + f32(1e-5)
    )
    return [f32(f32(v * inverse) * g) for v, g in zip(values, gamma)]


def projected(heads):
    step = 1 / (heads * 128)
    active = [f32(-0.5 + i * step) for i in range(2 * heads * 64)]
    return active + [math.nan] * (heads * 64)


def qk(values, heads, base, gamma):
    output = []
    step = f32(-2.0 * math.log(1000000.0) / 64)
    for row in range(2):
        for head in range(heads):
            start = (row * heads + head) * 64
            norm = normalized(values[start : start + 64], gamma)
            for lane in range(64):
                pair = lane % 32
                log_frequency = f32(pair * step)
                frequency = f32(math.exp(log_frequency))
                angle = f32((base + row) * frequency)
                low, high = norm[pair], norm[pair + 32]
                value = (
                    low * math.cos(angle) - high * math.sin(angle)
                    if lane < 32
                    else low * math.sin(angle) + high * math.cos(angle)
                )
                output.append(f32(value))
    return output + [0.0] * (heads * 64)


def history_key(logical, head, lane):
    return half(((logical + head * 3 + lane) % 17 - 8) / 64)


def history_value(logical, head, lane):
    return half((logical % 13 - 6) / 8 + (head % 5) / 16 + (lane % 7) / 32)


def main():
    gamma = [f32(0.875 + lane / 512) for lane in range(64)]
    q = projected(32)
    k = projected(8)
    v = [f32(-0.25 + i / 4096) for i in range(1024)] + [math.nan] * 512
    banks = {}
    for base in [0, 63, 1023]:
        banks[f"q.expected.{base}"] = qk(q, 32, base, gamma)
    q_norm = banks["q.expected.63"]
    k_norm = qk(k, 8, 63, gamma)
    banks["k.expected.63"] = k_norm
    scale = f32(1 / math.sqrt(64))
    attention = []
    for row in range(2):
        position = 63 + row
        for head in range(32):
            kv_head = head // 4
            query = q_norm[(row * 32 + head) * 64 : (row * 32 + head + 1) * 64]
            keys, values = [], []
            for logical in range(position + 1):
                if logical < 63:
                    keys.append(
                        [history_key(logical, kv_head, lane) for lane in range(64)]
                    )
                    values.append(
                        [history_value(logical, kv_head, lane) for lane in range(64)]
                    )
                else:
                    start = ((logical - 63) * 8 + kv_head) * 64
                    keys.append([half(x) for x in k_norm[start : start + 64]])
                    values.append([half(x) for x in v[start : start + 64]])
            scores = [
                sum(float(a) * b for a, b in zip(query, key)) * scale for key in keys
            ]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            attention.extend(
                f32(
                    sum(w * value[lane] for w, value in zip(weights, values))
                    / denominator
                )
                for lane in range(64)
            )
    # The extra per-element absolute1e-5 assertion is sufficient to prove
    # relativeL2<=2e-3 on this fixed active vector: ||expected||/sqrt(M)>=.005.
    # This strengthens the declared budget without a bespoke norm checker.
    rms_expected = math.sqrt(sum(v * v for v in attention) / len(attention))
    assert rms_expected >= 0.005, rms_expected
    banks["attention.expected"] = attention + [0.0] * 2048
    lines = ["// Generated independent F64 head/rotation/causal attention anchors."]
    for name, data in banks.items():
        encoded = b"".join(struct.pack("<f", v) for v in data).hex()
        lines.append(
            f'global.rodata.def @decoder.h64.attention.oracle.{name} = bytes("{encoded}")'
        )
    Path(sys.argv[1]).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
