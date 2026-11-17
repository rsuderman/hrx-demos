# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Check independent row completeness and the explicitly selected local anchors."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import os
import site
import sys
import math
import subprocess
import tempfile

sys.dont_write_bytecode = True
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
from reference import PINNED, load_adapter


def transport_request(row, name):
    request = json.loads(json.dumps(row["request"], ensure_ascii=False))
    request.setdefault("model", name)
    if "add_generation_prompt" not in row.get("kwargs", {}):
        options = request.setdefault("model_options", {})
        assert "add_generation_prompt" not in options, row["case"]
        options["add_generation_prompt"] = False
    return request


def source_error_message(row):
    message = row["error"]
    kind = row.get("error_type")
    # The Gemma oracle records the Python exception class in the same field;
    # other adapters record it separately. Native raise_exception preserves
    # the source message, not the Python class name.
    for prefix in ("TemplateError: ", "UndefinedError: "):
        if message.startswith(prefix):
            kind, message = prefix[:-2], message[len(prefix) :]
            break
    if kind == "UndefinedError" and message == "list object has no element 0":
        # The finite JSON helper deliberately has no Python Undefined object
        # provenance. Dereferencing the missing element calls jinja.check.
        return "invalid template value or operation"
    assert message, row["case"]
    return message


def compare_prompt(report, row, tokenizer):
    actual = report["prompt"]
    expected = row["text"]
    if "deferred_float" in row:
        metadata = row["deferred_float"]
        prefix, suffix = metadata["prefix"], metadata["suffix"]

        def split(text):
            assert prefix and suffix and text.count(prefix) == 1
            start = text.index(prefix) + len(prefix)
            end = text.index(suffix, start)
            number = float(text[start:end])
            assert math.isfinite(number)
            return text[:start], number, math.copysign(1, number), text[end:]

        assert split(actual) == split(expected), row["case"]
    else:
        assert actual == expected, row["case"]
    independent_ids = tokenizer.encode(actual, add_special_tokens=False).ids
    assert report["tokens"] == independent_ids, row["case"]
    if actual == expected:
        assert report["tokens"] == row["token_ids"], row["case"]
    return independent_ids


def check_packages(args):
    """Exercise existing exported cases through the public pinned Serve owner."""
    assert args.native_checker and args.form and args.native_output
    assert not args.adapter and not args.reference and not args.native_deferred
    assert {name: importlib.metadata.version(name) for name in PINNED} == PINNED
    import tokenizers

    assert (
        Path(tokenizers.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    )
    entries = json.loads(args.package_manifest.read_text())
    assert entries
    failures, records = [], []
    scratch_parent = os.environ["TEST_TMPDIR"]
    with tempfile.TemporaryDirectory(
        prefix="a401-reference-", dir=scratch_parent
    ) as scratch:
        root = Path(scratch)
        for entry in entries:
            control = Path(
                entry["source_output" if args.form == "source" else "bytecode_output"]
            )
            assert control.is_file(), control
            tokenizer_path = Path(entry["tokenizer"])
            tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
            rows = json.loads(Path(entry["reference"]).read_text())
            assert len(rows) == entry["expected_rows"]
            assert (
                sum("error" in row for row in rows) == entry["expected_template_errors"]
            )
            assert (
                sum("native_admission_error" in row for row in rows)
                == entry["expected_host_rejections"]
            )
            assert (
                sum("deferred_float" in row for row in rows)
                == entry["expected_deferred_rows"]
            )
            assert all(row["revision"] == entry["id"] for row in rows)
            by_case = {}
            for row in rows:
                by_case.setdefault(row["case"], []).append(row)

            def invoke(
                label,
                request,
                expected_error=None,
                response=None,
                chunk=0,
                truncated=False,
            ):
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request, ensure_ascii=False))
                command = [
                    str(args.native_checker),
                    "--control=" + str(control),
                    "--tokenizer=" + str(tokenizer_path),
                    "--request=" + str(request_path),
                ]
                if response is not None:
                    response_path = root / "response.txt"
                    response_path.write_text(response)
                    command += [
                        "--response=" + str(response_path),
                        "--response_chunk_size=" + str(chunk),
                        "--response_truncated=" + ("true" if truncated else "false"),
                    ]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=120
                )
                record = {
                    "id": entry["id"] + ":" + label,
                    "form": args.form,
                    "command": command,
                    "exit": result.returncode,
                    "stderr": result.stderr,
                }
                records.append(record)
                if expected_error is not None:
                    assert result.returncode == 1, (label, result.returncode)
                    assert not result.stdout, (label, result.stdout)
                    assert result.stderr, label
                    assert expected_error in result.stderr, (label, result.stderr)
                    record["expected_rejection"] = True
                    return None
                assert result.returncode == 0, (label, result.stderr)
                report = json.loads(result.stdout)
                assert report["model"] == entry["package_name"], label
                record["actual"] = report
                return report

            for index, row in enumerate(rows):
                label = "row:" + str(index) + ":" + row["case"]
                try:
                    error = (
                        row.get("native_admission_error")
                        if "native_admission_error" in row
                        else source_error_message(row) if "error" in row else None
                    )
                    report = invoke(
                        label, transport_request(row, entry["package_name"]), error
                    )
                    if report is not None:
                        ids = compare_prompt(report, row, tokenizer)
                        if "deferred_float" in row:
                            records[-1].update(
                                actual_text_hex=report["prompt"].encode().hex(),
                                token_ids=ids,
                                prepared_token_ids=report["tokens"],
                            )
                except (AssertionError, ValueError, subprocess.TimeoutExpired) as error:
                    failures.append((entry["id"], label, str(error)))

            for check in entry["checks"]:
                label = "check:" + check["id"]
                try:
                    request = json.loads(check["request"])
                    request.setdefault("model", entry["package_name"])
                    report = invoke(
                        label,
                        request,
                        check.get("error_contains"),
                        check.get("response"),
                        check.get("response_chunk_size", 0),
                        check.get("response_truncated", False),
                    )
                    if report is None:
                        continue
                    if "reference_case" in check:
                        if "reference_index" in check:
                            row = rows[check["reference_index"]]
                            assert row["case"] == check["reference_case"]
                        else:
                            candidates = by_case[check["reference_case"]]
                            assert len(candidates) == 1, check
                            row = candidates[0]
                        compare_prompt(report, row, tokenizer)
                    for key in (
                        "content",
                        "reasoning_content",
                        "checkpoint",
                        "reusable",
                        "tool_calls",
                    ):
                        expected = "expected_" + key
                        if expected in check:
                            assert report[key] == check[expected], (
                                label,
                                key,
                                report[key],
                            )
                except (AssertionError, ValueError, subprocess.TimeoutExpired) as error:
                    failures.append((entry["id"], label, str(error)))
    args.native_output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    assert not failures, failures
    print(f"PASS: {args.form} {len(records)} actual fixed-pin package checks")


p = argparse.ArgumentParser()
p.add_argument("--adapter", type=Path)
p.add_argument("--reference", type=Path)
p.add_argument("--source-override", action="store_true")
p.add_argument("--native-deferred", type=Path)
p.add_argument("--tokenizer", type=Path)
p.add_argument("--package-manifest", type=Path)
p.add_argument("--native-checker", type=Path)
p.add_argument("--form", choices=("source", "bytecode"))
p.add_argument("--native-output", type=Path)
a = p.parse_args()
if a.package_manifest:
    check_packages(a)
    raise SystemExit(0)
assert a.adapter and a.reference
assert not a.native_checker and not a.form and not a.native_output
adapter = load_adapter(a.adapter)
rows = json.loads(a.reference.read_text())
assert rows
for row in rows:
    assert row["revision"] == adapter.REVISION
    if "error" not in row:
        assert isinstance(row["token_ids"], list), "complete token vector required"
        assert all(isinstance(token, int) and token >= 0 for token in row["token_ids"])
        assert row["utf8_hex"] == row["text"].encode().hex()
        assert row["bytes"] == len(row["text"].encode())
adapter.check_reference(rows, a.source_override)
if a.native_deferred:
    assert a.tokenizer is not None, "--tokenizer required for actual native IDs"
    assert {name: importlib.metadata.version(name) for name in PINNED} == PINNED
    import tokenizers

    assert (
        Path(tokenizers.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(a.tokenizer))
    expected = {
        row["revision"] + ":" + row["case"] for row in rows if "deferred_float" in row
    }
    records = [json.loads(line) for line in a.native_deferred.read_text().splitlines()]
    # One component run may contain several profiles; each selected profile must
    # have its complete unique row x serialized-form set, including unchanged bytes.
    selected = [
        record for record in records if record["id"].startswith(adapter.REVISION + ":")
    ]
    assert len(selected) == 2 * len(expected)
    assert {(record["id"], record["form"]) for record in selected} == {
        (key, form) for key in expected for form in ("source", "bytecode")
    }
    for record in selected:
        actual_text = bytes.fromhex(record["actual_text_hex"]).decode("utf-8")
        independent_ids = tokenizer.encode(actual_text, add_special_tokens=False).ids
        assert record["token_ids"] == independent_ids, record["id"]
        assert record["prepared_token_ids"] == independent_ids, record["id"]
    print(
        f"PASS: {adapter.REVISION} {len(selected)} actual native/prepared HF ID records"
    )
else:
    assert a.tokenizer is None, "--tokenizer needs --native-deferred"
print(f"PASS: {adapter.REVISION} {len(rows)} independent oracle rows")
