# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Execute exact local template revisions through pinned HF/Jinja test helpers.

The task-owned venv owns dependencies. All checkpoint/tokenizer files are explicit
immutable external inputs; this script downloads nothing and constructs no model.
Outputs are independent oracle data, not outputs of the Loom provider under test.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import site
import sys

PINNED = {"transformers": "4.57.1", "Jinja2": "3.1.6", "tokenizers": "0.22.2", "MarkupSafe": "3.0.4"}
REVISIONS = {
    "7f0e5290": ("qwen35", "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"),
    "c3cf9e34": ("qwen35", "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "5816fce1": ("llama", "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    for revision in REVISIONS:
        p.add_argument("--tokenizer-" + revision, type=Path)
        p.add_argument("--template-" + revision, type=Path,
                       help="Actual mutated GGUF template bytes; independent oracle input")
    a = p.parse_args()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    assert versions == PINNED, versions
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from tokenizers import Tokenizer
    import transformers, jinja2, tokenizers
    for module in (transformers, jinja2, tokenizers):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    cases_path = a.source_root / "lib/text/qwen35/testing/render_cases.json"
    cases = json.loads(cases_path.read_text())
    rows = []
    tokenizers_by_revision = {}
    tokenizer_identities = {}
    for revision in REVISIONS:
        path = getattr(a, "tokenizer_" + revision)
        if path:
            tokenizers_by_revision[revision] = Tokenizer.from_file(str(path))
            tokenizer_identities[revision] = {"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    template_identities = {}
    for revision, (family, identity) in REVISIONS.items():
        path = a.source_root / "lib/text" / family / "templates" / (revision + ".jinja")
        override = getattr(a, "template_" + revision)
        if override:
            path = override
        source = path.read_bytes()
        actual_identity = hashlib.sha256(source).hexdigest()
        if not override:
            assert actual_identity == identity
        template_identities[revision] = {"path": str(path.resolve()),
            "sha256": actual_identity, "canonical_skeleton_sha256": identity,
            "override": bool(override)}
        template = _compile_jinja_template(source.decode())
        for case in cases:
            if revision not in case["revisions"]:
                continue
            request = copy.deepcopy(case["request"])
            # The native OpenAI chat transport decodes JSON argument strings
            # before invoking render_tool; normalize exactly this protocol seam.
            for message in request["messages"]:
                for call in message.get("tool_calls", []):
                    function = call.get("function", call)
                    if isinstance(function.get("arguments"), str):
                        function["arguments"] = json.loads(function["arguments"])
            kwargs = {"add_generation_prompt": True}
            kwargs.update(request.get("model_options", {}))
            if "enable_thinking" in request:
                kwargs["enable_thinking"] = request["enable_thinking"]
            if case.get("omit_generation_prompt"):
                kwargs.pop("add_generation_prompt", None)
            tools = request.get("tools")
            if revision == "5816fce1":
                # Explicit existing Llama baseline, not its tools/date branches.
                assert not tools and not request.get("enable_thinking", False)
                tools = None
                kwargs.update(bos_token="<|begin_of_text|>")
                if not override:
                    kwargs.update(date_string="26 Jul 2024")
            row = {"case": case["id"], "revision": revision, "template_sha256": actual_identity,
                   "request": case["request"], "kwargs": kwargs, "tools": tools}
            try:
                render_kwargs = dict(kwargs)
                if revision == "5816fce1" and override:
                    # The supported baseline uses its source fallback, not wall time.
                    # Do not inject the expected date or use the production extractor.
                    render_kwargs["strftime_now"] = jinja2.Undefined(name="strftime_now")
                rendered = template.render(messages=request["messages"], tools=tools, **render_kwargs)
            except Exception as error:
                if not case.get("error"):
                    raise
                row.update(error_type=type(error).__name__, error=str(error))
            else:
                assert not case.get("error"), (revision, case["id"], "expected template error")
                data = rendered.encode()
                row.update(text=rendered, utf8_hex=data.hex(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                if revision in tokenizers_by_revision:
                    row["token_ids"] = tokenizers_by_revision[revision].encode(rendered, add_special_tokens=False).ids
            rows.append(row)
    assert not a.output.exists(), "oracle output must be a new directory"
    a.output.mkdir(parents=True)
    (a.output / "reference.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    helper_path = Path(inspect.getsourcefile(inspect.unwrap(_compile_jinja_template)))
    manifest = {"versions": versions, "python": sys.version, "interpreter": sys.executable,
                "helper": str(helper_path), "helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(), "rows": len(rows),
                "token_vector_rows": sum("token_ids" in row for row in rows),
                "tokenizers": tokenizer_identities, "templates": template_identities, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "independent reference only; no native Loom parity claimed"}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
