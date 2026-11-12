# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent pinned-HF oracle with explicit local adapter and immutable inputs."""
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

PINNED = {"transformers": "4.57.1", "Jinja2": "3.1.6",
          "tokenizers": "0.22.2", "MarkupSafe": "3.0.4"}


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("local_reference_adapter", path)
    assert spec is not None and spec.loader is not None, path
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def identity(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def main():
    p = argparse.ArgumentParser()
    for flag in ("adapter", "cases", "template", "tokenizer", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--source-override", action="store_true",
                   help="Explicit original/mutated GGUF template input")
    a = p.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    assert versions == PINNED, versions
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from tokenizers import Tokenizer
    import transformers, jinja2, tokenizers
    for module in (transformers, jinja2, tokenizers):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    adapter = load_adapter(a.adapter)
    source = a.template.read_bytes()
    actual_hash = hashlib.sha256(source).hexdigest()
    if not a.source_override:
        assert actual_hash == adapter.TEMPLATE_SHA256, "canonical template hash mismatch"
    template = _compile_jinja_template(source.decode())
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    cases = json.loads(a.cases.read_text())
    assert cases, "case input must be nonempty"
    rows = []
    for case in cases:
        # A case file may cover multiple revisions owned by this local suite.
        if adapter.REVISION not in case["revisions"]:
            continue
        original_request = copy.deepcopy(case["request"])
        result = adapter.render_kwargs(case, a.source_override)
        assert isinstance(result, tuple) and len(result) == 3
        messages, tools, kwargs = result
        assert case["request"] == original_request, "adapter mutated the oracle request"
        # Jinja Undefined denotes an absent input (e.g. source date fallback).
        # Preserve the original JSON kwargs while passing that absence to Jinja.
        recorded_kwargs = {key: value for key, value in kwargs.items()
                           if not isinstance(value, jinja2.Undefined)}
        row = {"case": case["id"], "revision": adapter.REVISION,
               "template_sha256": actual_hash, "request": original_request,
               "kwargs": recorded_kwargs, "tools": tools}
        for key in ("native_admission_error", "native_counterpart", "template_case"):
            if key in case:
                row[key] = copy.deepcopy(case[key])
        try:
            rendered = template.render(messages=messages, tools=tools, **kwargs)
        except Exception as error:
            if not case.get("error"):
                raise
            row.update(adapter.error_record(error))
        else:
            assert not case.get("error"), (adapter.REVISION, case["id"], "expected template error")
            data = rendered.encode()
            row.update(text=rendered, utf8_hex=data.hex(), bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest(),
                       token_ids=tokenizer.encode(rendered, add_special_tokens=False).ids)
        rows.append(row)
    assert rows, "no cases selected for explicitly supplied adapter"
    adapter.check_reference(rows, a.source_override)
    assert not a.output.exists(), "oracle output must be a new directory"
    a.output.mkdir(parents=True)
    (a.output / "reference.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    helper = Path(inspect.getsourcefile(inspect.unwrap(_compile_jinja_template)))
    manifest = {"versions": versions, "python": sys.version, "interpreter": sys.executable,
                "user_site": site.ENABLE_USER_SITE, "revision": adapter.REVISION,
                "helper": identity(helper), "adapter": identity(a.adapter),
                "cases": identity(a.cases), "tokenizer": identity(a.tokenizer),
                "template": {**identity(a.template),
                             "canonical_skeleton_sha256": adapter.TEMPLATE_SHA256,
                             "override": a.source_override},
                "script": identity(Path(__file__)), "rows": len(rows),
                "token_vector_rows": sum("token_ids" in row for row in rows),
                "scope": "independent reference only; no native Loom parity claimed"}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
