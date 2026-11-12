# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent scalar expectations for the stated scaled-half matrix policy.

No production math or native runtime is imported. Matrix accumulation uses F64
on exact staged-half operands and checks the fixed F32 ordering-error budget.
"""

import math
import struct
from fixtures import f16, f32, stage_weight


def prepare(values):
    values = [f32(x) for x in values]
    maximum = max(map(abs, values), default=0.0)
    biased = (struct.unpack("<I", struct.pack("<f", maximum))[0] >> 23) & 255
    exponent = max(0, min(100, biased - 127 - 13))
    scale = math.ldexp(1.0, -exponent)
    return [f16(f32(x * scale)) for x in values], scale, math.ldexp(1.0, exponent)


def decode_staged(weight, tag, input_width, output_width):
    block_bytes = {12: 144, 14: 210}[tag]
    row_bytes = input_width // 256 * block_bytes
    assert len(weight) == row_bytes * output_width
    decoded = []
    for column in range(output_width):
        row = []
        for block_index in range(input_width // 256):
            offset = column * row_bytes + block_index * block_bytes
            block = weight[offset : offset + block_bytes]
            row.extend(stage_weight(block, tag, k) for k in range(256))
        decoded.append(row)
    return decoded


def scaled_matrix(inputs, weight, tag, output_width):
    input_width = len(inputs[0])
    decoded = decode_staged(weight, tag, input_width, output_width)
    prepared = [prepare(row) for row in inputs]
    return [
        [
            f32(f32(math.fsum(float(x) * w for x, w in zip(values, row))) * inverse)
            for row in decoded
        ]
        for values, _, inverse in prepared
    ]


def silu(value):
    # Stable independent evaluation, including large negative finite gates.
    if value >= 0:
        return value / (1 + math.exp(-value))
    e = math.exp(value)
    return value * e / (1 + e)


def gated_matrix(inputs, gate, up, output_width):
    a = scaled_matrix(inputs, gate, 12, output_width)
    b = scaled_matrix(inputs, up, 12, output_width)
    return [[f32(f32(silu(x)) * y) for x, y in zip(g, u)] for g, u in zip(a, b)]


def rms(values, gamma, epsilon):
    # Scalar/F64 reduction independent of subgroup/workgroup traversal.
    mean = f32(math.fsum(float(f32(x * x)) for x in values) / len(values))
    inverse = f32(1 / math.sqrt(f32(mean + epsilon)))
    return [f32(f32(x * inverse) * w) for x, w in zip(values, gamma)]


def neox(values, position, log_step):
    width = len(values)
    assert width % 2 == 0
    output = [0.0] * width
    for k in range(width // 2):
        frequency = f32(math.exp(f32(k * log_step)))
        angle = f32(position * frequency)
        cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
        low, high = values[k], values[k + width // 2]
        output[k] = f32(f32(low * cosine) - f32(high * sine))
        output[k + width // 2] = f32(f32(high * cosine) + f32(low * sine))
    return output
