# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local independent template inputs; never imports a production recognizer."""

import copy
import json

REVISION = "5816fce1"
TEMPLATE_SHA256 = "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    kwargs = {"add_generation_prompt": True}
    kwargs.update(request.get("model_options", {}))
    if "enable_thinking" in request:
        kwargs["enable_thinking"] = request["enable_thinking"]
    if case.get("omit_generation_prompt"):
        kwargs.pop("add_generation_prompt", None)
    tools = request.get("tools")
    kwargs["bos_token"] = "<|begin_of_text|>"
    if source_override:
        import jinja2

        # Use the template's own source fallback, never wall time or extraction.
        kwargs["strftime_now"] = jinja2.Undefined(name="strftime_now")
    else:
        kwargs["date_string"] = "26 Jul 2024"
    return (request["messages"], tools, kwargs)


def error_record(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def check_reference(rows, source_override: bool):
    assert len(rows) == 11
    assert sum("error" in row for row in rows) == 0
    assert sum("token_ids" in row for row in rows) == 11
    r = {row["case"]: row for row in rows}
    assert "Environment: ipython\n" in r["empty-tools"]["text"]
    assert "Environment: ipython\n" not in r["one-user"]["text"]
    assert r["unused-thinking"]["text"] == r["one-user"]["text"]
    assert (
        "<|start_header_id|>not-a-role<|end_header_id|>\n\nlate<|eot_id|>"
        in r["source-custom-role"]["text"]
    )
    generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assert r["one-user"]["text"].endswith(generation_prefix)
    assert (
        r["generation-false"]["text"]
        == r["one-user"]["text"][: -len(generation_prefix)]
    )
    for row in rows:
        assert row["text"].count("<|begin_of_text|>") == 1
        if not source_override:
            assert "Today Date: 26 Jul 2024\n\n" in row["text"]
        assert row["token_ids"][0] == 128000
        assert row["token_ids"].count(128000) == 1
