#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent operand/policy FFN and live Qwen gate/up numerical witnesses."""

import importlib.util
import math
from pathlib import Path
import sys

helper, mode, destination = sys.argv[1:]
spec = importlib.util.spec_from_file_location("fixture_helpers", helper)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.out.clear()
h.emit("// Copyright 2026 AMD ROCm Software\n// SPDX-License-Identifier: MIT\n")
for tag, name in (
    (2, "q4_0"),
    (3, "q4_1"),
    (6, "q5_0"),
    (8, "q8_0"),
    (12, "q4k"),
    (13, "q5k"),
    (14, "q6k"),
    (23, "iq4_xs"),
):
    h.emit(
        f"config.def @ggml.linear_{name}_q8_1_x4.token_capacity = 512 : index\nconfig.def @ggml.linear_{name}_q8_1_x4.output_capacity = 262144 : index\n"
    )
h.emit("config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n")


def extent(tag, width, columns):
    return (
        width
        * columns
        // (32 if tag in (2, 3, 6, 8) else 256)
        * {2: 18, 3: 20, 6: 22, 8: 34, 12: 144, 13: 176, 14: 210}[tag]
    )


def weight_body(tag, width, role, buffer, label, up_divisor=8187):
    # Reuse the maintained independent K-quant encoders. Change only fixture
    # scales to expose compatible MMQ half-folding vs canonical F32 scales.
    divisor = up_divisor if role == "up" else 4093
    if tag in (12, 13, 14):
        body = h.weight_body(tag, width, role, buffer, label)
        body = body.replace(
            f"scalar.constant {8192 if role == 'up' else 4096}.0",
            f"scalar.constant {divisor}.0",
        )
        body = body.replace("scalar.constant 1 : i8", "scalar.constant 3 : i8")
        return body
    size = {2: 18, 3: 20, 6: 22, 8: 34}[tag]
    factor = "%inverse_channel" if role == "up" else "%channel_factor"
    body = f"""  %blocks_{label} = index.constant {width//32} : index
  %stride_{label} = index.constant {size} : offset
  %rowbytes_{label} = index.constant {width//32*size} : offset
  %row_{label} = index.scale %channel, %rowbytes_{label} : index, offset -> offset
  %factor32_{label} = index.cast {factor} : index to i32
  %factor_{label} = scalar.uitofp %factor32_{label} : i32 to f32
  %divisor_{label} = scalar.constant {divisor}.0 : f32
  %d32_{label} = scalar.divf %factor_{label}, %divisor_{label} : f32
  %d16_{label} = scalar.fptrunc %d32_{label} : f32 to f16
  scf.for %b_{label} = [%zero to %blocks_{label} step %one] {{
    %add_{label} = index.scale %b_{label}, %stride_{label} : index, offset -> offset
    %base_{label} = index.add %row_{label}, %add_{label} : offset
    %ds_{label} = buffer.view %{buffer}[%base_{label}] : buffer -> view<1xf16>
    view.store %d16_{label}, %ds_{label}[0] : f16, view<1xf16>
    %bytes_{label} = buffer.view %{buffer}[%base_{label}] : buffer -> view<{size}xi8>
"""
    if tag == 3:
        # q=2, m=d produces the independently assumed logical coefficient3*d.
        # Keep nonzero affine data through the entire normalize/gate/up/down path.
        body += f"""    %madd_{label} = index.constant 2 : offset
    %morigin_{label} = index.add %base_{label}, %madd_{label} : offset
    %mview_{label} = buffer.view %{buffer}[%morigin_{label}] : buffer -> view<1xf16>
    view.store %d16_{label}, %mview_{label}[0] : f16, view<1xf16>
"""
    payloads = {
        2: ((2, 18, -69),),
        3: ((4, 20, 34),),
        6: ((2, 6, -1), (6, 22, 51)),
        8: ((2, 34, 3),),
    }
    for number, (begin, end, code) in enumerate(payloads[tag]):
        body += f"""    %lo_{label}{number} = index.constant {begin} : index
    %hi_{label}{number} = index.constant {end} : index
    %code_{label}{number} = scalar.constant {code} : i8
    scf.for %j_{label}{number} = [%lo_{label}{number} to %hi_{label}{number} step %one] {{
      view.store %code_{label}{number}, %bytes_{label}[%j_{label}{number}] : i8, view<{size}xi8>
    }}
"""
    return body + "  }\n"


def policy(profile, tag):
    return 0 if profile == 0 and tag in (12, 14) else 1


def coefficient(tag, role, channel, selected_policy, active, up_divisor=8187):
    factor = 4 - channel if role == "up" else channel + 1
    d = h.f16(h.f32(factor / (up_divisor if role == "up" else 4093)))
    combined = h.f32(3 * d)
    # Only retained Q4 MMQ rounds the combined weight scale to half.
    return (
        h.f16(combined)
        if tag == 12 and selected_policy == 0 and active > 8
        else combined
    )


def quantized(values, selected_policy, active):
    maximum = max(map(abs, values), default=0.0)
    if not maximum:
        return [0] * len(values), 0.0
    if selected_policy == 0 and active > 8:
        inverse = h.f32(127 / maximum)
        d = h.f32(1 / inverse)
    else:
        d = (
            h.f32(maximum * h.f32(1 / 127))
            if selected_policy == 0
            else h.f32(maximum / 127)
        )
        inverse = h.f32(1 / d)
    scaled = [h.f32(v * inverse) for v in values]
    # Keep activation-code choices robust to retained native reciprocal/exp
    # estimates. The independent oracle never reads actual packed bytes.
    assert min(abs(abs(v - round(v)) - 0.5) for v in scaled) > 0.001
    codes = [math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5) for v in scaled]
    assert min(codes) >= -127 and max(codes) <= 127
    return codes, h.f16(d)


def silu(value):
    if value >= 0:
        return value / (1 + math.exp(-value))
    e = math.exp(value)
    return value * e / (1 + e)


def gate_up(row, hidden, gt, ut, profile, active, normalized=False):
    # Each direction needs its own discriminating data: the subsequent Q8
    # pack can erase a gate difference for some otherwise valid up scales.
    up_divisor = 8191 if (gt, ut) == (12, 8) else 8187
    initial = (-1 if row % 2 else 1) * (row + 1)
    x = 0.9921875 * (row + 1)
    if normalized:
        value = h.f32(initial / math.sqrt(initial * initial + 1e-6) * (127 / 32))
        codes, d = quantized([value] * 4, policy(profile, gt), active)
        xg = codes[0] * d
        codes, d = quantized([value] * 4, policy(profile, ut), active)
        xu = codes[0] * d
    else:
        xg = xu = x
    gates = [
        h.f32(xg * hidden * coefficient(gt, "gate", c, policy(profile, gt), active))
        for c in range(4)
    ]
    ups = [
        h.f32(
            xu
            * hidden
            * coefficient(ut, "up", c, policy(profile, ut), active, up_divisor)
        )
        for c in range(4)
    ]
    return [h.f32(h.f32(silu(g)) * u) for g, u in zip(gates, ups)]


def final_values(row, hidden, ffn, tags, profile, active):
    gt, ut, dt = tags
    activated = gate_up(row, hidden, gt, ut, profile, active, True)
    codes, d = quantized(activated, policy(profile, dt), active)
    total = ffn // 4 * sum(codes)
    initial = (-1 if row % 2 else 1) * (row + 1)
    return [
        h.f32(
            initial
            + total * d * coefficient(dt, "down", c, policy(profile, dt), active)
        )
        for c in range(4)
    ]


def readonly_checks(buffers):
    return "".join(
        f"      check.expect.bitwise actual(%{name}_owner) expected(%{name}_owner) : tensor<{size+guards}x{kind}>\n"
        for name, size, kind, guards in buffers
    )


def classic():
    h.emit(
        "config.def @llama.model.hidden_size = 1024 : index\nconfig.def @llama.model.rms_epsilon = 0.000001 : f32\n"
    )
    # The reused normalization adapter also exports an unused public layer
    # wrapper. Satisfy that wrapper's strict-link facts without replacing it.
    h.emit(
        "config.def @loomify.ffn.profile = 0 : index\nfunc.def public pure inline @loomify.config.layer_ffn(%layer:index) -> (index,index,index,index,index,index,index) {\n  %F = index.constant 256 : index\n  %q4 = index.constant 12 : index\n  %q6 = index.constant 14 : index\n  %gb = index.constant 147456 : index\n  %db = index.constant 215040 : index\n  func.return %F,%q4,%q4,%q6,%gb,%gb,%db : index,index,index,index,index,index,index\n}\n"
    )
    h.emit(
        "template.decl @loomify.ffn(%E:index,%F:index,%T:index,%phase:index,%profile:index,%epsilon:f32,%gt:index,%ut:index,%dt:index,%gb:index,%ub:index,%db:index,%residual:buffer,%count:buffer,%gamma:view<[%E]xf32>,%gate:view<[%gb]xi8>,%up:view<[%ub]xi8>,%down:view<[%db]xi8>)\n"
    )
    hidden, ffn = 1024, 256
    tuples = [
        (0, (12, 12, 14)),
        (0, (12, 14, 14)),
        (0, (8, 8, 8)),
        (0, (8, 12, 14)),
        (0, (12, 8, 8)),
        (1, (12, 13, 14)),
        (1, (13, 12, 14)),
        (1, (14, 12, 8)),
    ]
    if mode == "classic_q4":
        tuples = [(0, (2, 2, 2)), (0, (2, 2, 3)), (1, (3, 2, 3))]
    for profile, tags in tuples:
        gt, ut, dt = tags
        cases = [(1, 1), (17, 0), (17, 1), (17, 17)]
        if (profile, tags) in ((0, (12, 12, 14)), (0, (8, 12, 14)), (1, (12, 13, 14))):
            cases += [(rows, active) for rows in (8, 16, 64) for active in (0, 1, rows)]
        for rows, active in cases:
            # The legacy decode route has an existing nonempty precondition.
            name = f"independent_p{profile}_{gt}_{ut}_{dt}_t{rows}_a{active}"
            gb, ub, db = (
                extent(gt, hidden, ffn),
                extent(ut, hidden, ffn),
                extent(dt, ffn, hidden),
            )
            init = (
                h.prefix()
                + f"  %F = index.constant {ffn} : index\n  %gate_row = index.cmp ult, %channel, %F : index\n  scf.if %gate_row {{\n"
            )
            up_divisor = 8191 if (gt, ut) == (12, 8) else 8187
            init += (
                weight_body(gt, hidden, "gate", "gate", "g")
                + weight_body(ut, hidden, "up", "up", "u", up_divisor)
                + "  }\n"
            )
            init += weight_body(dt, ffn, "down", "down", "d")
            init += f"  %gamma_v = buffer.view %gamma[%origin] : buffer -> view<{hidden}xf32>\n  %gamma_f = scalar.constant 3.96875 : f32\n  view.store %gamma_f,%gamma_v[%channel] : f32,view<{hidden}xf32>\n"
            init += f"  %residual_v = buffer.view %residual[%origin] : buffer -> view<{rows}x{hidden}xf32>\n  %nanbits = scalar.constant 2143289344 : i32\n  %nan = scalar.bitcast %nanbits : i32 to f32\n"
            for row in range(rows):
                value = (-1 if row % 2 else 1) * (row + 1)
                init += f"  %r{row} = scalar.constant {float(value)} : f32\n  view.store {'%r'+str(row) if row<active else '%nan'},%residual_v[{row},%channel] : f32,view<{rows}x{hidden}xf32>\n"
            h.kernel(
                name + ".init",
                hidden,
                "%gate:buffer,%up:buffer,%down:buffer,%gamma:buffer,%residual:buffer",
                init,
            )
            h.reference(
                name + ".init",
                hidden,
                "%gate:buffer,%up:buffer,%down:buffer,%gamma:buffer,%residual:buffer",
                f"    func.call @{name}.init.body(%channel,%gate,%up,%down,%gamma,%residual) : (index,buffer,buffer,buffer,buffer,buffer) -> ()",
            )
            (
                h.kernel(
                    "test.expected." + name,
                    hidden,
                    "%residual:buffer",
                    h.prefix()
                    + h.store_pattern(
                        active,
                        hidden,
                        "residual",
                        [
                            final_values(r, hidden, ffn, tags, profile, active)
                            for r in range(active)
                        ],
                        "expected",
                    ),
                )
                if active
                else None
            )
            if active:
                h.reference(
                    name,
                    hidden,
                    "%gate:buffer,%up:buffer,%down:buffer,%gamma:buffer,%residual:buffer,%count:buffer",
                    f"    func.call @test.expected.{name}.body(%channel,%residual) : (index,buffer) -> ()",
                )
            else:
                h.emit(
                    f"func.def @reference.{name}(%gate:buffer,%up:buffer,%down:buffer,%gamma:buffer,%residual:buffer,%count:buffer) {{ func.return }}\n"
                )
            h.emit(
                f"""command.def @{name}.actual() launch(%gate:buffer,%up:buffer,%down:buffer,%gamma:buffer,%residual:buffer,%count:buffer) {{
  %E = index.constant {hidden} : index
  %F = index.constant {ffn} : index
  %T = index.constant {rows} : index
  %phase = index.constant {1 if rows == 1 else 0} : index
  %profile = index.constant {profile} : index
  %gt = index.constant {gt} : index
  %ut = index.constant {ut} : index
  %dt = index.constant {dt} : index
  %gb = index.constant {gb} : index
  %ub = index.constant {ub} : index
  %db = index.constant {db} : index
  %origin = index.constant 0 : offset
  %epsilon = scalar.constant 0.000001 : f32
  %g = buffer.view %gate[%origin] : buffer -> view<[%gb]xi8>
  %u = buffer.view %up[%origin] : buffer -> view<[%ub]xi8>
  %d = buffer.view %down[%origin] : buffer -> view<[%db]xi8>
  %gamma_view = buffer.view %gamma[%origin] : buffer -> view<[%E]xf32>
  template.apply<@loomify.ffn>(%E,%F,%T,%phase,%profile,%epsilon,%gt,%ut,%dt,%gb,%ub,%db,%residual,%count,%gamma_view,%g,%u,%d) : (index,index,index,index,index,f32,index,index,index,index,index,index,buffer,buffer,view<[%E]xf32>,view<[%gb]xi8>,view<[%ub]xi8>,view<[%db]xi8>)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
"""
            )
            readonly = [
                ("gate", gb, "i8", 512),
                ("up", ub, "i8", 512),
                ("down", db, "i8", 512),
                ("gamma", hidden, "f32", 8),
            ]
            for buffer, size, kind, guards in readonly + [
                ("residual", rows * hidden, "f32", 8)
            ]:
                offset = 256 if kind == "i8" else 16
                fill = -23 if kind == "i8" else -99.0
                h.emit(
                    f"    %{buffer}_owner = check.generate.fill value({fill}) : tensor<{size+guards}x{kind}>\n    %{buffer} = check.tensor.view %{buffer}_owner offset({offset}) : tensor<{size+guards}x{kind}> -> tensor<{size}x{kind}>\n"
                )
            h.emit(
                f"    %prefix = check.tensor.view %residual_owner offset(0) : tensor<{rows*hidden+8}xf32> -> tensor<4xf32>\n    %suffix = check.tensor.view %residual_owner offset({(rows*hidden+4)*4}) : tensor<{rows*hidden+8}xf32> -> tensor<4xf32>\n    %count = check.generate.fill value({active}) : tensor<3xi32>\n    check.generate<@reference.{name}.init>(%gate,%up,%down,%gamma,%residual) : (tensor<{gb}xi8>,tensor<{ub}xi8>,tensor<{db}xi8>,tensor<{hidden}xf32>,tensor<{rows*hidden}xf32>) -> ()\n"
            )
            if active < rows:
                h.emit(
                    f"    %inactive = check.tensor.view %residual offset({active*hidden*4}) : tensor<{rows*hidden}xf32> -> tensor<{(rows-active)*hidden}xf32>\n"
                )
            types = f"tensor<{gb}xi8>,tensor<{ub}xi8>,tensor<{db}xi8>,tensor<{hidden}xf32>,tensor<{rows*hidden}xf32>,tensor<3xi32>"
            h.emit(
                f"    check.compare<@{name}.actual,@reference.{name}>(%gate,%up,%down,%gamma,%residual,%count) : ({types}) -> () {{\n      check.expect.approximate actual(%residual_owner) expected(%residual_owner) atol(0.0001) rtol(0.0001) nan(same) : tensor<{rows*hidden+8}xf32>,tensor<{rows*hidden+8}xf32>\n      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>\n      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>\n      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>\n"
                + readonly_checks(readonly)
            )
            if active < rows:
                h.emit(
                    f"      check.expect.bitwise actual(%inactive) expected(%inactive) : tensor<{(rows-active)*hidden}xf32>\n"
                )
            h.emit("    }\n  }\n  check.return\n}\n")


def live_qwen():
    hidden, ffn, bank = 1024, 256, 2097152
    words, groups = bank // 4, bank // 4 // 256
    pairs = ((6, 6), (8, 8), (6, 13))
    for key, value in (
        ("hidden_size", hidden),
        ("ffn_size", ffn),
        ("projection_implementation", 1),
        ("ffn_row_bytes", ffn * 4),
        ("ffn_capacity_bytes", 512 * ffn * 4),
        ("hidden_q8_capacity_bytes", 512 * hidden // 128 * 144),
        ("hidden_q8_depth4_bytes", 4 * hidden // 128 * 144),
    ):
        h.emit(f"config.def @qwen.model.{key} = {value} : index\n")
    h.emit(
        "command.decl @qwen38_ffn_gate_up_prefill(%T:index,%S:index,%layer:index,%depth:index,%spans:index) launch(%root:buffer,%input:buffer,%count:buffer,%output:buffer)\n"
    )
    h.emit(
        "func.def public pure inline @loomify.config.layer_ffn(%layer:index) -> (index,index,index,index,index,index,index) {\n  %zero = index.constant 0 : index\n  %one = index.constant 1 : index\n  %first = index.cmp eq,%layer,%zero : index\n  %second = index.cmp eq,%layer,%one : index\n"
    )
    fields = [
        (ffn, ffn, ffn),
        (6, 8, 6),
        (6, 8, 13),
        (14, 14, 14),
        tuple(extent(gt, hidden, ffn) for gt, _ in pairs),
        tuple(extent(ut, hidden, ffn) for _, ut in pairs),
        (extent(14, ffn, hidden),) * 3,
    ]
    for i, values in enumerate(fields):
        for j, value in enumerate(values):
            h.emit(f"  %f{i}_{j} = index.constant {value} : index\n")
        h.emit(
            f"  %later{i} = scf.select %second,%f{i}_1,%f{i}_2 : index\n  %field{i} = scf.select %first,%f{i}_0,%later{i} : index\n"
        )
    h.emit(
        "  func.return "
        + ",".join(f"%field{i}" for i in range(7))
        + " : "
        + ",".join(["index"] * 7)
        + "\n}\n"
    )
    # Integer-only before/after comparison proves public-command parameter
    # reads without relying on the test harness's opaque parameter placement.
    for action in ("snapshot", "unchanged"):
        args = "%root:buffer,%before:buffer" + (
            ",%flags:buffer" if action == "unchanged" else ""
        )
        h.emit(f"""kernel.def @live_qwen.{action}() {{
  %groups = index.constant {groups} : index
  %one = index.constant 1 : index
  %threads = index.constant 256 : index
  kernel.launch.config workgroups(%groups,%one,%one) workgroup_size(%threads,%one,%one) : index
}} launch({args}) {{
  %group = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %threads = index.constant 256 : index
  %base = index.mul %group,%threads : index
  %i = index.add %base,%lane : index
  %origin = index.constant 0 : offset
  %root_v = buffer.view %root[%origin] : buffer -> view<{words}xi32>
  %before_v = buffer.view %before[%origin] : buffer -> view<{words}xi32>
  %actual = view.load %root_v[%i] : view<{words}xi32> -> i32
""")
        if action == "snapshot":
            h.emit(f"  view.store %actual,%before_v[%i] : i32,view<{words}xi32>\n")
        else:
            h.emit(f"""  %expected = view.load %before_v[%i] : view<{words}xi32> -> i32
  %different = scalar.cmpi ne,%actual,%expected : i32
  %flag = scalar.extui %different : i1 to i32
  %sum = kernel.workgroup.reduce<addi> %flag : i32
  %zero = index.constant 0 : index
  %first = index.cmp eq,%lane,%zero : index
  scf.if %first {{
    %flags_v = buffer.view %flags[%origin] : buffer -> view<{groups}xi32>
    view.store %sum,%flags_v[%group] : i32,view<{groups}xi32>
  }}
""")
        h.emit("  kernel.return\n}\n")
    for layer, (gt, ut) in enumerate(pairs):
        gb, ub = extent(gt, hidden, ffn), extent(ut, hidden, ffn)
        h.kernel(
            f"live_qwen.weights{layer}",
            ffn,
            "%gate:buffer,%up:buffer",
            h.prefix()
            + weight_body(gt, hidden, "gate", "gate", "g")
            + weight_body(ut, hidden, "up", "up", "u"),
        )
        for rows in (1, 3, 16, 17, 64):
            for active in sorted({0, 1, rows}):
                name = f"live_qwen_{gt}_{ut}_t{rows}_a{active}"
                inactive_view = (
                    f"    %inactive = check.tensor.view %output offset({active*ffn*4}) : tensor<{512*ffn}xf32> -> tensor<{(rows-active)*ffn}xf32>\n"
                    if active < rows
                    else ""
                )
                inactive_check = (
                    f"      check.expect.bitwise actual(%inactive) expected(%inactive) : tensor<{(rows-active)*ffn}xf32>\n"
                    if active < rows
                    else ""
                )
                init = (
                    h.prefix()
                    + f"  %x = buffer.view %input[%origin] : buffer -> view<{rows}x{hidden}xf32>\n  %bits = scalar.constant 2143289344 : i32\n  %nan = scalar.bitcast %bits : i32 to f32\n"
                )
                for row in range(rows):
                    init += f"  %v{row} = scalar.constant {h.fp(0.9921875*(row+1))} : f32\n  view.store {'%v'+str(row) if row<active else '%nan'},%x[{row},%channel] : f32,view<{rows}x{hidden}xf32>\n"
                h.kernel(name + ".input", hidden, "%input:buffer", init)
                h.reference(
                    name + ".input",
                    hidden,
                    "%input:buffer",
                    f"    func.call @{name}.input.body(%channel,%input) : (index,buffer) -> ()",
                )
                expected = h.prefix() + h.store_pattern(
                    rows,
                    ffn,
                    "output",
                    [
                        (
                            gate_up(r, hidden, gt, ut, 1, active)
                            if r < active
                            else [0.0] * 4
                        )
                        for r in range(rows)
                    ],
                    "answer",
                )
                h.kernel("test.expected." + name, ffn, "%output:buffer", expected)
                h.reference(
                    name,
                    ffn,
                    "%root:buffer,%input:buffer,%count:buffer,%output:buffer,%flags:buffer",
                    f"    func.call @test.expected.{name}.body(%channel,%output) : (index,buffer) -> ()",
                )
                # Separate flags oracle: expected command leaves root opaque,
                # but requires zero mismatches in every 256-word block.
                h.emit(
                    f"""func.def @{name}.expected(%root:buffer,%input:buffer,%count:buffer,%output:buffer,%flags:buffer) {{
  func.call @reference.{name}(%root,%input,%count,%output,%flags) : (buffer,buffer,buffer,buffer,buffer) -> ()
  %origin = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %end = index.constant {groups} : index
  %z = scalar.constant 0 : i32
  %f = buffer.view %flags[%origin] : buffer -> view<{groups}xi32>
  scf.for %i = [%zero to %end step %one] {{ view.store %z,%f[%i] : i32,view<{groups}xi32> }}
  func.return
}}
command.def @{name}.actual() launch(%root:buffer,%input:buffer,%count:buffer,%output:buffer,%flags:buffer) {{
  %layer = index.constant {layer} : index
  %T = index.constant {rows} : index
  %zero = index.constant 0 : index
  %gb = index.constant {gb} : index
  %ub = index.constant {ub} : index
  %gate = command.parameter %root,"blk.{{}}.ffn_gate.weight"[%layer] : view<[%gb]xi8>
  %up = command.parameter %root,"blk.{{}}.ffn_up.weight"[%layer] : view<[%ub]xi8>
  kernel.launch @live_qwen.weights{layer}[](%gate,%up) : [](view<[%gb]xi8>,view<[%ub]xi8>)
  %bytes = index.constant {bank} : offset
  %before = buffer.alloca<global> align(256) %bytes : buffer
  kernel.launch @live_qwen.snapshot[](%root,%before) : [](buffer,buffer)
  command.launch @qwen38_ffn_gate_up_prefill[%T,%T,%layer,%zero,%zero](%root,%input,%count,%output) : [index,index,index,index,index](buffer,buffer,buffer,buffer)
  kernel.launch @live_qwen.unchanged[](%root,%before,%flags) : [](buffer,buffer,buffer)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
    %root_owner = check.generate.fill value(77) : tensor<{bank+512}xi8>
    %root = check.tensor.view %root_owner offset(256) : tensor<{bank+512}xi8> -> tensor<{bank}xi8>
    %root_prefix = check.tensor.view %root_owner offset(0) : tensor<{bank+512}xi8> -> tensor<256xi8>
    %root_suffix = check.tensor.view %root_owner offset({bank+256}) : tensor<{bank+512}xi8> -> tensor<256xi8>
    %input_owner = check.generate.fill value(-777.0) : tensor<{rows*hidden+8}xf32>
    %input = check.tensor.view %input_owner offset(16) : tensor<{rows*hidden+8}xf32> -> tensor<{rows*hidden}xf32>
    check.generate<@reference.{name}.input>(%input) : (tensor<{rows*hidden}xf32>) -> ()
    %count = check.generate.fill value({active}) : tensor<3xi32>
    %owner = check.generate.fill value(-99.0) : tensor<{512*ffn+8}xf32>
    %output = check.tensor.view %owner offset(16) : tensor<{512*ffn+8}xf32> -> tensor<{512*ffn}xf32>
    %prefix = check.tensor.view %owner offset(0) : tensor<{512*ffn+8}xf32> -> tensor<4xf32>
    %suffix = check.tensor.view %owner offset({(512*ffn+4)*4}) : tensor<{512*ffn+8}xf32> -> tensor<4xf32>
    %tail = check.tensor.view %output offset({rows*ffn*4}) : tensor<{512*ffn}xf32> -> tensor<{(512-rows)*ffn}xf32>
    %flags = check.generate.fill value(-1) : tensor<{groups}xi32>
{inactive_view}
    check.compare<@{name}.actual,@{name}.expected>(%root,%input,%count,%output,%flags) : (tensor<{bank}xi8>,tensor<{rows*hidden}xf32>,tensor<3xi32>,tensor<{512*ffn}xf32>,tensor<{groups}xi32>) -> () {{
      check.expect.approximate actual(%owner) expected(%owner) atol(0.0001) rtol(0.0001) nan(same) : tensor<{512*ffn+8}xf32>,tensor<{512*ffn+8}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>
      check.expect.bitwise actual(%tail) expected(%tail) : tensor<{(512-rows)*ffn}xf32>
      check.expect.bitwise actual(%root_prefix) expected(%root_prefix) : tensor<256xi8>
      check.expect.bitwise actual(%root_suffix) expected(%root_suffix) : tensor<256xi8>
      check.expect.bitwise actual(%flags) expected(%flags) : tensor<{groups}xi32>
      check.expect.bitwise actual(%input_owner) expected(%input_owner) : tensor<{rows*hidden+8}xf32>
      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>
{inactive_check}
    }}
  }}
  check.return
}}
"""
                )


def mixed_matrix(tiny=False):
    hidden, ffn = 1024, 256
    h.emit(
        "template.decl @llm.projection.gated_matrix(%T:index,%E:index,%F:index,%gt:index,%ut:index,%gb:index,%ub:index,%count:buffer,%input:buffer,%gate:view<[%gb]xi8>,%up:view<[%ub]xi8>,%output:buffer)\n"
    )
    pairs = ((12, 14),) if tiny else ((12, 14), (13, 12))
    cases = (
        ((1, 1),)
        if tiny
        else ((1, 1), (17, 0), (17, 1), (17, 17), (64, 0), (64, 1), (64, 64))
    )
    for gt, ut in pairs:
        gb, ub = extent(gt, hidden, ffn), extent(ut, hidden, ffn)
        for rows, active in cases:
            padded = rows if rows <= 16 else (rows + 63) // 64 * 64
            name = f"mixed_matrix_{gt}_{ut}_t{rows}_a{active}"
            init = (
                h.prefix()
                + f"  %F = index.constant {ffn} : index\n  %weight_row = index.cmp ult,%channel,%F : index\n  scf.if %weight_row {{\n"
            )
            # Weights stage exactly in F16. Input values are dyadic but rows
            # beyond16 can require an extra mantissa bit, so each operand's
            # independent oracle must retain its selected preparation route.
            init += weight_body(gt, hidden, "gate", "gate", "g").replace(
                "scalar.constant 4093.0", "scalar.constant 4096.0"
            )
            init += weight_body(ut, hidden, "up", "up", "u", 8192) + "  }\n"
            init += f"  %x = buffer.view %input[%origin] : buffer -> view<{rows}x{hidden}xf32>\n  %bits = scalar.constant 2143289344 : i32\n  %nan = scalar.bitcast %bits : i32 to f32\n"
            for row in range(rows):
                init += f"  %v{row} = scalar.constant {h.fp(0.9921875*(row+1))} : f32\n  view.store {'%v'+str(row) if row<active else '%nan'},%x[{row},%channel] : f32,view<{rows}x{hidden}xf32>\n"
            h.kernel(
                name + ".init", hidden, "%input:buffer,%gate:buffer,%up:buffer", init
            )
            h.reference(
                name + ".init",
                hidden,
                "%input:buffer,%gate:buffer,%up:buffer",
                f"    func.call @{name}.init.body(%channel,%input,%gate,%up) : (index,buffer,buffer,buffer) -> ()",
            )
            anchors = []
            for row in range(rows):
                x = 0.9921875 * (row + 1)
                gx = h.f16(x) if rows > 16 and gt in (12, 14) else x
                ux = h.f16(x) if rows > 16 and ut in (12, 14) else x
                anchors.append(
                    [
                        h.f32(
                            h.f32(silu(h.f32(gx * hidden * 3 * (c + 1) / 4096)))
                            * h.f32(ux * hidden * 3 * (4 - c) / 8192)
                        )
                        for c in range(4)
                    ]
                    if row < active
                    else [0.0] * 4
                )
            h.kernel(
                "test.expected." + name,
                ffn,
                "%output:buffer",
                h.prefix() + h.store_pattern(rows, ffn, "output", anchors, "answer"),
            )
            h.reference(
                name,
                ffn,
                "%input:buffer,%gate:buffer,%up:buffer,%count:buffer,%output:buffer",
                f"    func.call @test.expected.{name}.body(%channel,%output) : (index,buffer) -> ()",
            )
            h.emit(
                f"""command.def @{name}.actual() launch(%input:buffer,%gate:buffer,%up:buffer,%count:buffer,%output:buffer) {{
  %T = index.constant {rows} : index
  %E = index.constant {hidden} : index
  %F = index.constant {ffn} : index
  %gt = index.constant {gt} : index
  %ut = index.constant {ut} : index
  %gb = index.constant {gb} : index
  %ub = index.constant {ub} : index
  %origin = index.constant 0 : offset
  %g = buffer.view %gate[%origin] : buffer -> view<[%gb]xi8>
  %u = buffer.view %up[%origin] : buffer -> view<[%ub]xi8>
  template.apply<@llm.projection.gated_matrix>(%T,%E,%F,%gt,%ut,%gb,%ub,%count,%input,%g,%u,%output) : (index,index,index,index,index,index,index,buffer,buffer,view<[%gb]xi8>,view<[%ub]xi8>,buffer)
  command.return
}}
check.scenario @{name}.witness {{
  check.trial[1](%trial:index,%entropy:check.entropy) {{
"""
            )
            readonly = [
                ("input", rows * hidden, "f32", 8),
                ("gate", gb, "i8", 512),
                ("up", ub, "i8", 512),
            ]
            for buffer, size, kind, guards in readonly:
                offset = 256 if kind == "i8" else 16
                fill = -23 if kind == "i8" else -777.0
                h.emit(
                    f"    %{buffer}_owner = check.generate.fill value({fill}) : tensor<{size+guards}x{kind}>\n    %{buffer} = check.tensor.view %{buffer}_owner offset({offset}) : tensor<{size+guards}x{kind}> -> tensor<{size}x{kind}>\n"
                )
            h.emit(
                f"    check.generate<@reference.{name}.init>(%input,%gate,%up) : (tensor<{rows*hidden}xf32>,tensor<{gb}xi8>,tensor<{ub}xi8>) -> ()\n    %count = check.generate.fill value({active}) : tensor<3xi32>\n    %owner = check.generate.fill value(-99.0) : tensor<{padded*ffn+8}xf32>\n    %output = check.tensor.view %owner offset(16) : tensor<{padded*ffn+8}xf32> -> tensor<{padded*ffn}xf32>\n    %prefix = check.tensor.view %owner offset(0) : tensor<{padded*ffn+8}xf32> -> tensor<4xf32>\n    %suffix = check.tensor.view %owner offset({(padded*ffn+4)*4}) : tensor<{padded*ffn+8}xf32> -> tensor<4xf32>\n"
            )
            tails = [
                (tail, start, size)
                for tail, start, size in (
                    ("inactive", active * ffn, (rows - active) * ffn),
                    ("padding", rows * ffn, (padded - rows) * ffn),
                )
                if size
            ]
            for tail, start, size in tails:
                h.emit(
                    f"    %{tail} = check.tensor.view %output offset({start*4}) : tensor<{padded*ffn}xf32> -> tensor<{size}xf32>\n"
                )
            h.emit(
                f"    check.compare<@{name}.actual,@reference.{name}>(%input,%gate,%up,%count,%output) : (tensor<{rows*hidden}xf32>,tensor<{gb}xi8>,tensor<{ub}xi8>,tensor<3xi32>,tensor<{padded*ffn}xf32>) -> () {{\n      check.expect.approximate actual(%owner) expected(%owner) atol(0.0001) rtol(0.0001) nan(same) : tensor<{padded*ffn+8}xf32>,tensor<{padded*ffn+8}xf32>\n      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<4xf32>\n      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<4xf32>\n      check.expect.bitwise actual(%count) expected(%count) : tensor<3xi32>\n"
                + readonly_checks(readonly)
            )
            for tail, _, size in tails:
                h.emit(
                    f"      check.expect.bitwise actual(%{tail}) expected(%{tail}) : tensor<{size}xf32>\n"
                )
            h.emit("    }\n  }\n  check.return\n}\n")


if mode in ("classic", "classic_q4"):
    classic()
elif mode == "live_qwen":
    live_qwen()
elif mode in ("gated_tiny", "gated_matrix"):
    mixed_matrix(mode == "gated_tiny")
else:
    raise ValueError(mode)
Path(destination).write_text("".join(h.out))
