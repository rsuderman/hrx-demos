# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Regenerate shared Unicode 16 casing deltas against the native 15.1 base.

Inputs are explicit, immutable copies under UCD_DIR/{15.1.0,16.0.0}/.
This developer tool neither downloads data nor uses Python's Unicode tables.
The optional base proof contains sparse *full* mappings: an omitted scalar maps
to itself. Compare those expectations against every scalar in the selected
native runtime before accepting generated product data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

INPUTS = {
    (
        "15.1.0",
        "UnicodeData.txt",
    ): "2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86",
    (
        "15.1.0",
        "SpecialCasing.txt",
    ): "55a477efd933a52cd27e6a9bf70265bb2d8814af31aab07767abc8eb421f27ef",
    (
        "16.0.0",
        "UnicodeData.txt",
    ): "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
    (
        "16.0.0",
        "SpecialCasing.txt",
    ): "8d5de354eef79f2395a54c9c7dcebbaf3d30fc962d0f85611ea97aa973a0c451",
    (
        "16.0.0",
        "DerivedCoreProperties.txt",
    ): "39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd",
}

# Retain the Unicode data copyright and permission notice with derived data.
# Source: https://www.unicode.org/license.txt (Unicode License V3).
UNICODE_LICENSE = """UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
"""


def read_inputs(root):
    result = {}
    for (version, name), expected in INPUTS.items():
        path = root / version / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"UCD input hash mismatch: {version}/{name}")
        text = raw.decode("utf-8")
        # UnicodeData is headerless. Its pinned bytes and versioned input
        # identity establish the version; the other files declare it too.
        if name != "UnicodeData.txt":
            header = f"# {Path(name).stem}-{version}.txt"
            if text.splitlines()[0] != header:
                raise ValueError(f"UCD version header mismatch: {version}/{name}")
        result[version, name] = text
    return result


def scalar(codepoint):
    return 0 <= codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF


def mapping(value):
    result = tuple(int(item, 16) for item in value.split())
    if not result or not all(scalar(item) for item in result):
        raise ValueError("invalid full case mapping")
    return result


def case_maps(inputs, version):
    lower, upper = {}, {}
    for line in inputs[version, "UnicodeData.txt"].splitlines():
        fields = line.split(";")
        if len(fields) != 15:
            raise ValueError("invalid UnicodeData row")
        codepoint = int(fields[0], 16)
        # First/Last ranges have identity casing. Surrogate ranges are not
        # scalars and have no mappings; they need no expansion here.
        if fields[12]:
            upper[codepoint] = mapping(fields[12])
        if fields[13]:
            lower[codepoint] = mapping(fields[13])
    final_sigma = False
    for line in inputs[version, "SpecialCasing.txt"].splitlines():
        data = line.partition("#")[0].strip()
        if not data:
            continue
        fields = [part.strip() for part in data.split(";")]
        codepoint = int(fields[0], 16)
        condition = fields[4]
        if not condition:
            lower[codepoint] = mapping(fields[1])
            upper[codepoint] = mapping(fields[3])
        elif condition == "Final_Sigma":
            if codepoint != 0x03A3 or mapping(fields[1]) != (0x03C2,):
                raise ValueError("unexpected contextual lowercase mapping")
            final_sigma = True
        elif condition.split()[0] not in ("lt", "tr", "az"):
            raise ValueError(f"unhandled default casing condition: {condition}")
    if not final_sigma:
        raise ValueError("missing Final_Sigma contract")
    for table, maximum in ((lower, 2), (upper, 3)):
        for codepoint, value in table.items():
            if not scalar(codepoint) or not 1 <= len(value) <= maximum:
                raise ValueError("mapping exceeds the selected native case ABI")
    return lower, upper


def property_ranges(inputs, name):
    ranges = []
    for line in inputs["16.0.0", "DerivedCoreProperties.txt"].splitlines():
        fields = [part.strip() for part in line.partition("#")[0].split(";")]
        if len(fields) != 2 or fields[1] != name:
            continue
        ends = fields[0].split("..")
        start, end = int(ends[0], 16), int(ends[-1], 16)
        if not scalar(start) or not scalar(end) or start > end:
            raise ValueError(f"invalid {name} range")
        if ranges and start <= ranges[-1][1]:
            raise ValueError(f"unordered {name} ranges")
        if ranges and start == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    if not ranges:
        raise ValueError(f"missing {name} property")
    return ranges


def delta_rows(base, target):
    rows = []
    # The union also includes changed mappings whose target became identity.
    for codepoint in sorted(base.keys() | target.keys()):
        value = target.get(codepoint, (codepoint,))
        if value != base.get(codepoint, (codepoint,)):
            rows.append((codepoint, len(value), *value, *((0,) * (3 - len(value)))))
    return rows


def rodata(name, rows):
    data = b"".join(struct.pack("<" + "I" * len(row), *row) for row in rows)
    return (
        f"global.constant @jinja.case_{name}_count : index = {len(rows)}\n\n"
        f'global.rodata.def @jinja.case_{name}_rows = align(4) bytes("{data.hex()}")\n'
    )


LOOKUPS = """
// One lower-bound lookup serves both case directions. A zero count means
// there is no override; an explicit identity override has count one.
func.def @jinja.case_delta(%storage: buffer, %count: index, %codepoint: i32) -> (i32, i32, i32, i32) {
  %origin = index.constant 0 : offset
  %rows = buffer.view %storage[%origin] : buffer -> view<[%count]x5xi32>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %position_value, %unused = scf.while(%low = %zero : index, %high = %count : index) -> (index, index) {
    %active = index.cmp ult, %low, %high : index
    scf.condition %active, %low, %high : i1, index, index
  } do(%low: index, %high: index) {
    %sum = index.add %low, %high : index
    %middle_value = index.div %sum, %two : index
    %middle = index.assume %middle_value [ge(%middle_value, 0), lt(%middle_value, %count)] : index
    %key = view.load %rows[%middle, 0] : view<[%count]x5xi32> -> i32
    %before = scalar.cmpi ult, %key, %codepoint : i32
    %after = index.add %middle, %one : index
    %next_low = scf.select %before, %after, %low : index
    %next_high = scf.select %before, %high, %middle : index
    scf.yield %next_low, %next_high : index, index
  }
  %position = index.assume %position_value [ge(%position_value, 0), le(%position_value, %count)] : index
  %present = index.cmp ult, %position, %count : index
  %empty = scalar.constant 0 : i32
  %length, %first, %second, %third = scf.if %present -> (i32, i32, i32, i32) {
    %key = view.load %rows[%position, 0] : view<[%count]x5xi32> -> i32
    %matches = scalar.cmpi eq, %key, %codepoint : i32
    %n = view.load %rows[%position, 1] : view<[%count]x5xi32> -> i32
    %a = view.load %rows[%position, 2] : view<[%count]x5xi32> -> i32
    %b = view.load %rows[%position, 3] : view<[%count]x5xi32> -> i32
    %c = view.load %rows[%position, 4] : view<[%count]x5xi32> -> i32
    %length = scf.select %matches, %n, %empty : i32
    scf.yield %length, %a, %b, %c : i32, i32, i32, i32
  } else {
    scf.yield %empty, %empty, %empty, %empty : i32, i32, i32, i32
  }
  func.return %length, %first, %second, %third : i32, i32, i32, i32
}

func.def public @jinja.case_override(%codepoint: i32, %upper: i1) -> (i32, i32, i32, i32) {
  %storage, %count = scf.if %upper -> (buffer, index) {
    %rows = global.load @jinja.case_upper_rows : buffer
    %count = global.load @jinja.case_upper_count : index
    scf.yield %rows, %count : buffer, index
  } else {
    %rows = global.load @jinja.case_lower_rows : buffer
    %count = global.load @jinja.case_lower_count : index
    scf.yield %rows, %count : buffer, index
  }
  %length, %a, %b, %c = func.call @jinja.case_delta(%storage, %count, %codepoint) : (buffer, index, i32) -> (i32, i32, i32, i32)
  func.return %length, %a, %b, %c : i32, i32, i32, i32
}

// One range lookup serves both Unicode16 contextual-casing properties.
func.def @jinja.case_in_ranges(%storage: buffer, %count: index, %codepoint: i32) -> (i1) {
  %origin = index.constant 0 : offset
  %rows = buffer.view %storage[%origin] : buffer -> view<[%count]x2xi32>
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %position_value, %unused = scf.while(%low = %zero : index, %high = %count : index) -> (index, index) {
    %active = index.cmp ult, %low, %high : index
    scf.condition %active, %low, %high : i1, index, index
  } do(%low: index, %high: index) {
    %sum = index.add %low, %high : index
    %middle_value = index.div %sum, %two : index
    %middle = index.assume %middle_value [ge(%middle_value, 0), lt(%middle_value, %count)] : index
    %end = view.load %rows[%middle, 1] : view<[%count]x2xi32> -> i32
    %before = scalar.cmpi ult, %end, %codepoint : i32
    %after = index.add %middle, %one : index
    %next_low = scf.select %before, %after, %low : index
    %next_high = scf.select %before, %high, %middle : index
    scf.yield %next_low, %next_high : index, index
  }
  %position = index.assume %position_value [ge(%position_value, 0), le(%position_value, %count)] : index
  %present = index.cmp ult, %position, %count : index
  %inside = scf.if %present -> (i1) {
    %start = view.load %rows[%position, 0] : view<[%count]x2xi32> -> i32
    %inside = scalar.cmpi uge, %codepoint, %start : i32
    scf.yield %inside : i1
  } else {
    %false = scalar.constant false : i1
    scf.yield %false : i1
  }
  func.return %inside : i1
}
"""


def generate(lower, upper, cased, ignorable):
    lines = [
        "// Copyright 2026 AMD ROCm Software",
        "// SPDX-License-Identifier: MIT AND Unicode-3.0",
        "// Generated by build_tools/generate_unicode_case_data.py; do not edit.",
        "// Native full mapping base: Unicode 15.1.0. Target: Unicode 16.0.0.",
        "// Regenerate with --ucd-dir containing the versioned inputs below.",
        "// Upstream input URL prefix: https://www.unicode.org/Public/",
    ]
    lines.extend(
        f"// {version}/ucd/{name} sha256={digest}"
        for (version, name), digest in INPUTS.items()
    )
    lines.extend(
        "// " + line if line else "//" for line in UNICODE_LICENSE.splitlines()
    )
    lines.append("")
    for name, rows in (
        ("lower", lower),
        ("upper", upper),
        ("cased", cased),
        ("ignorable", ignorable),
    ):
        lines.append(rodata(name, rows))
    lines.append(LOOKUPS.strip())
    for name in ("cased", "ignorable"):
        lines.append(f"""
func.def public @jinja.case_{name}(%codepoint: i32) -> (i1) {{
  %storage = global.load @jinja.case_{name}_rows : buffer
  %count = global.load @jinja.case_{name}_count : index
  %inside = func.call @jinja.case_in_ranges(%storage, %count, %codepoint) : (buffer, index, i32) -> (i1)
  func.return %inside : i1
}}""")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucd-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-proof-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    inputs = read_inputs(args.ucd_dir)
    base_lower, base_upper = case_maps(inputs, "15.1.0")
    target_lower, target_upper = case_maps(inputs, "16.0.0")
    lower = delta_rows(base_lower, target_lower)
    upper = delta_rows(base_upper, target_upper)
    cased = property_ranges(inputs, "Cased")
    ignorable = property_ranges(inputs, "Case_Ignorable")
    result = generate(lower, upper, cased, ignorable)
    args.output.write_text(result, encoding="utf-8")
    if args.base_proof_output:
        # Each row is [codepoint, count, first, second, third]. Unused results
        # are zero; native agreement compares only the declared result count.
        proof = {"unicode_version": "15.1.0", "identity_default": True}
        for name, table in (("lower", base_lower), ("upper", base_upper)):
            proof[name] = [
                [cp, len(value), *value, *((0,) * (3 - len(value)))]
                for cp, value in sorted(table.items())
                if value != (cp,)
            ]
        args.base_proof_output.write_text(json.dumps(proof) + "\n", encoding="utf-8")
    if args.summary_output:
        summary = {
            "base_unicode_version": "15.1.0",
            "target_unicode_version": "16.0.0",
            "inputs": {
                f"{version}/{name}": digest
                for (version, name), digest in INPUTS.items()
            },
            "lower_delta_rows": len(lower),
            "upper_delta_rows": len(upper),
            "cased_ranges": len(cased),
            "case_ignorable_ranges": len(ignorable),
            "packed_data_bytes": 20 * (len(lower) + len(upper))
            + 8 * (len(cased) + len(ignorable)),
            "output_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
        }
        args.summary_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
