#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent scalar math anchors; no production Loom or GPU calls."""
import math
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def normalized(values, gamma):
    inverse = 1.0 / math.sqrt(sum(float(v) ** 2 for v in values) / len(values) + f32(1e-6))
    return [f32(f32(v * inverse) * g) for v, g in zip(values, gamma)]


def main():
    banks = {}
    gamma = [f32(0.75 + i / 2048) for i in range(1024)]
    rows = [[f32(-1 + (i + r * 1024) / 1024) for i in range(1024)] for r in range(2)]
    banks['rms.input'] = sum(rows, []) + [math.nan] * 1024
    banks['rms.gamma'] = gamma
    banks['rms.expected'] = sum([normalized(row, gamma) for row in rows], []) + [0.0] * 1024
    banks['last.expected'] = normalized(rows[1], gamma)
    banks['residual.expected'] = [f32(v + 7.0) for v in sum(rows, [])] + [7.0] * 1024
    lines = ['// Generated independent F64 anchors and explicit F32 stage stores.']
    for name, values in banks.items():
        encoded = b''.join(struct.pack('<f', v) for v in values).hex()
        lines.append(f'global.rodata.def @decoder.oracle.{name} = bytes("{encoded}")')
    Path(sys.argv[1]).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
