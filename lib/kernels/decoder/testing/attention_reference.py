#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent head RMS/NeoX and F64 causal GQA anchors at actual geometry."""
import math
from pathlib import Path
import struct
import sys
from reference import f32, half, normalized


def projected(heads):
    step = 1 / (heads * 256)
    active = [f32(-0.5 + i * step) for i in range(2 * heads * 128)]
    return active + [math.nan] * (heads * 128)


def qk(values, heads, base, gamma):
    output = []
    step = f32(-2.0 * math.log(1000000.0) / 128)
    for row in range(2):
        for head in range(heads):
            start = (row * heads + head) * 128
            norm = normalized(values[start:start + 128], gamma)
            for lane in range(128):
                pair = lane % 64
                log_frequency = f32(pair * step)
                frequency = f32(math.exp(log_frequency))
                angle = f32((base + row) * frequency)
                low, high = norm[pair], norm[pair + 64]
                value = low * math.cos(angle) - high * math.sin(angle) if lane < 64 else low * math.sin(angle) + high * math.cos(angle)
                output.append(f32(value))
    return output + [0.0] * (heads * 128)


def history_key(logical, head, lane):
    return half(((logical + head * 3 + lane) % 17 - 8) / 64)


def history_value(logical, head, lane):
    return half((logical % 13 - 6) / 8 + (head % 5) / 16 + (lane % 7) / 32)


def main():
    gamma = [f32(0.875 + lane / 512) for lane in range(128)]
    q = projected(16)
    k = projected(8)
    v = [f32(-0.25 + i / 4096) for i in range(2048)] + [math.nan] * 1024
    banks = {}
    for base in [0, 63, 1023]:
        banks[f'q.expected.{base}'] = qk(q, 16, base, gamma)
    q_norm = banks['q.expected.63']
    k_norm = qk(k, 8, 63, gamma)
    banks['k.expected.63'] = k_norm
    scale = f32(1 / math.sqrt(128))
    attention = []
    for row in range(2):
        position = 63 + row
        for head in range(16):
            kv_head = head // 2
            query = q_norm[(row * 16 + head) * 128:(row * 16 + head + 1) * 128]
            keys, values = [], []
            for logical in range(position + 1):
                if logical < 63:
                    keys.append([history_key(logical, kv_head, lane) for lane in range(128)])
                    values.append([history_value(logical, kv_head, lane) for lane in range(128)])
                else:
                    start = ((logical - 63) * 8 + kv_head) * 128
                    keys.append([half(x) for x in k_norm[start:start + 128]])
                    values.append([half(x) for x in v[start:start + 128]])
            scores = [sum(float(a) * b for a, b in zip(query, key)) * scale for key in keys]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            attention.extend(f32(sum(w * value[lane] for w, value in zip(weights, values)) / denominator) for lane in range(128))
    # The extra per-element absolute1e-5 assertion is sufficient to prove
    # relativeL2<=2e-3 on this fixed active vector: ||expected||/sqrt(M)>=.005.
    # This strengthens the declared budget without a bespoke norm checker.
    rms_expected = math.sqrt(sum(v * v for v in attention) / len(attention))
    assert rms_expected >= 0.005, rms_expected
    banks['attention.expected'] = attention + [0.0] * 2048
    lines = ['// Generated independent F64 head/rotation/causal attention anchors.']
    for name, data in banks.items():
        encoded = b''.join(struct.pack('<f', v) for v in data).hex()
        lines.append(f'global.rodata.def @decoder.attention.oracle.{name} = bytes("{encoded}")')
    Path(sys.argv[1]).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
