# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent original-Jinja oracle for raw compiler and package fixtures."""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import site
import sys

sys.dont_write_bytecode = True

PINNED = {
    "transformers": "4.57.1",
    "Jinja2": "3.1.6",
    "tokenizers": "0.22.2",
    "MarkupSafe": "3.0.4",
}


def identity(path):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("local_reference_adapter", path)
    assert spec is not None and spec.loader is not None, path
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def typed_json(value):
    """Preserve mapping order, numeric types, and signed zero in raw fixtures."""
    if isinstance(value, dict):
        return ("object", [(key, typed_json(item)) for key, item in value.items()])
    if isinstance(value, list):
        return ("array", [typed_json(item) for item in value])
    if isinstance(value, float):
        return ("float", value.hex())
    return (type(value).__name__, value)


def main():
    parser = argparse.ArgumentParser()
    for name in ("cases", "template", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--check-native", type=Path)
    parser.add_argument(
        "--source-override",
        action="store_true",
        help="Explicit original/mutated GGUF template input for an adapter",
    )
    args = parser.parse_args()
    assert not args.source_override or args.adapter
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    versions = {key: importlib.metadata.version(key) for key in PINNED}
    assert versions == PINNED, versions
    from transformers.utils.chat_template_utils import _compile_jinja_template
    import transformers
    import jinja2
    import tokenizers

    for module in (transformers, jinja2, tokenizers):
        assert (
            Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    if args.check_native:
        references = json.loads((args.output / "reference.json").read_text())
        expected = {row["id"] for row in references if "deferred_float" in row}
        actual = [
            json.loads(line) for line in args.check_native.read_text().splitlines()
        ]
        assert len(actual) == 2 * len(expected)
        assert {(row["id"], row["form"]) for row in actual} == {
            (key, form) for key in expected for form in ("source", "bytecode")
        }
        for row in actual:
            text = bytes.fromhex(row["actual_text_hex"]).decode("utf-8")
            assert (
                tokenizer.encode(text, add_special_tokens=False).ids == row["token_ids"]
            ), row
        print(
            json.dumps(
                {"native_deferred_rows": len(actual), "independent_token_ids": "PASS"}
            )
        )
        return
    original_bytes = args.template.read_bytes()
    original = original_bytes.decode("utf-8")
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    adapter = load_adapter(args.adapter) if args.adapter else None
    if adapter and not args.source_override:
        assert (
            original_hash == adapter.TEMPLATE_SHA256
        ), "canonical template hash mismatch"
    cases = json.loads(args.cases.read_text())
    assert cases
    if not adapter:
        assert len({case["id"] for case in cases}) == len(cases)

    def bind_supplied_date(binding):
        # Source keyword binding is format=; fixture state stays in the closure.
        def supplied_date(format):
            if (
                not isinstance(binding.get("format"), str)
                or format != binding["format"]
            ):
                raise jinja2.TemplateError("unsupported supplied date format")
            result = binding.get("result")
            if not isinstance(result, str):
                raise jinja2.TemplateError("supplied date result must be a string")
            return result

        return supplied_date

    rows = []
    for case_index, case in enumerate(cases):
        if adapter and adapter.REVISION not in case.get("revisions", []):
            continue
        source = case.get("source", original)
        if "template_sha256" in case:
            assert original_hash == case["template_sha256"]
        for mutation in case.get("mutations", []):
            count = mutation.get("occurrences", 1)
            assert source.count(mutation["find"]) == count, case["id"]
            source = source.replace(mutation["find"], mutation["replace"], count)
        if adapter:
            request = copy.deepcopy(case["request"])
            result = adapter.render_kwargs(case, args.source_override)
            assert isinstance(result, tuple) and len(result) == 3
            messages, tools, kwargs = result
            assert case["request"] == request, "adapter changed source request"
            recorded_kwargs = {
                key: value
                for key, value in kwargs.items()
                if not isinstance(value, jinja2.Undefined)
            }
            explicit_environment = dict(recorded_kwargs, messages=messages, tools=tools)
            environment = dict(kwargs, messages=messages, tools=tools)
        else:
            explicit_environment = copy.deepcopy(case["environment"])
            environment = copy.deepcopy(explicit_environment)
            # Raw compiler fixtures carry data, never a wall-clock callable.
            # Adapter fixtures retain their own explicit Undefined/default policy.
            environment["strftime_now"] = jinja2.Undefined(name="strftime_now")
            capability = environment.pop("__loomify_strftime_now", None)
            if isinstance(capability, dict):
                environment["strftime_now"] = bind_supplied_date(capability)
        row = {
            "id": (
                f"{adapter.REVISION}:{case_index}:{case['id']}"
                if adapter
                else case["id"]
            ),
            "case": case["id"],
            "source": source,
            "environment": explicit_environment,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        if "environment_source" in case:
            assert not adapter
            assert typed_json(json.loads(case["environment_source"])) == typed_json(
                explicit_environment
            )
            row["environment_source"] = case["environment_source"]
        if "deferred_float" in case:
            assert not case.get("native_unsupported")
            row["deferred_float"] = copy.deepcopy(case["deferred_float"])
        if adapter:
            row.update(
                revision=adapter.REVISION,
                template_sha256=original_hash,
                request=request,
                kwargs=recorded_kwargs,
                tools=tools,
            )
        for key in ("native_admission_error", "native_counterpart", "template_case"):
            if key in case:
                row[key] = copy.deepcopy(case[key])
        if case.get("native_unsupported"):
            # Keep the independent Jinja result, while explicitly recording a
            # reviewed native capability limit. This is not a template error.
            row["native_unsupported"] = case["native_unsupported"]
        template = _compile_jinja_template(source)
        try:
            rendered = template.render(**environment)
        except Exception as error:
            assert case.get("error") or case.get("error_contains"), (
                case["id"],
                str(error),
            )
            if case.get("error_contains"):
                assert case["error_contains"] in str(error), (case["id"], str(error))
            row.update(
                error_contains=case.get("error_contains", ""), oracle_error=str(error)
            )
            if adapter:
                row.update(adapter.error_record(error))
            # A source raise_exception message must survive native execution.
            # Undefined/type diagnostics may legitimately have a native wrapper.
            if type(error) is jinja2.exceptions.TemplateError:
                row["native_error_contains"] = str(error)
        else:
            assert not case.get("error") and not case.get("error_contains"), case["id"]
            data = rendered.encode("utf-8")
            row.update(
                text=rendered,
                utf8_hex=data.hex(),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                token_ids=tokenizer.encode(rendered, add_special_tokens=False).ids,
            )
        if "native_error_contains" in case:
            row["native_error_contains"] = case["native_error_contains"]
        rows.append(row)
    assert rows, "no selected cases"
    if adapter:
        adapter.check_reference(rows, args.source_override)
    assert not args.output.exists(), "output must be a fresh directory"
    args.output.mkdir(parents=True)
    (args.output / "reference.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = {
        "versions": versions,
        "python": sys.version,
        "interpreter": sys.executable,
        "user_site": site.ENABLE_USER_SITE,
        "revision": adapter.REVISION if adapter else None,
        "helper": identity(
            Path(inspect.getsourcefile(inspect.unwrap(_compile_jinja_template)))
        ),
        "cases": identity(args.cases),
        "template": {
            **identity(args.template),
            "original_template_sha256": (
                adapter.TEMPLATE_SHA256 if adapter else original_hash
            ),
            "override": args.source_override,
        },
        "tokenizer": identity(args.tokenizer),
        "script": identity(Path(__file__)),
        "adapter": identity(args.adapter) if adapter else None,
        "rows": len(rows),
        "oracle_success_rows": sum("text" in row for row in rows),
        "token_vector_rows": sum("token_ids" in row for row in rows),
        "exact_rows": sum(
            "text" in row
            and "native_unsupported" not in row
            and "deferred_float" not in row
            for row in rows
        ),
        "deferred_float_rows": sum("deferred_float" in row for row in rows),
        "error_rows": sum("error_contains" in row for row in rows),
        "native_unsupported_rows": sum("native_unsupported" in row for row in rows),
        "scope": "independent Jinja reference; no native execution claimed",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
