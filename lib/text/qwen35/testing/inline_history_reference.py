# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local independent template inputs; never imports a production recognizer."""

import copy
import json

REVISION = "7f0e5290"
TEMPLATE_SHA256 = "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    # Native transport parses JSON argument strings before XML rendering.
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
    return (request["messages"], tools, kwargs)


def error_record(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def check_reference(rows, source_override: bool):
    assert len(rows) == 34
    assert sum("error" in row for row in rows) == 6
    assert sum("token_ids" in row for row in rows) == 28
    r = {row["case"]: row for row in rows}
    assert (
        r["tool-arguments-float-exponent"]["text"]
        == r["tool-arguments-float-decimal"]["text"]
    )
    assert (
        "<parameter=n>\n1.0\n</parameter>" in r["tool-arguments-float-exponent"]["text"]
    )
    assert (
        "<parameter=n>\n-0.0\n</parameter>" in r["tool-arguments-negative-zero"]["text"]
    )
    assert "café <>&\n" in r["tool-arguments-types"]["text"]
    assert "<parameter=n>\n1\n</parameter>" in r["tool-arguments-types-integer"]["text"]
    assert r["one-user"]["text"].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert "<parameter=yes>\nTrue\n</parameter>" in r["tool-arguments-types"]["text"]
    assert "<parameter=none>\nNone\n</parameter>" in r["tool-arguments-types"]["text"]
    assert "old thought" not in r["assistant-history"]["text"]
    assert "new thought" in r["assistant-history"]["text"]
