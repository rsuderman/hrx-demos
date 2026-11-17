# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local independent template inputs; never imports a production recognizer."""

import copy
import json

REVISION = "c3cf9e34"
TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


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
    assert len(rows) == 42
    assert sum("error" in row for row in rows) == 7
    assert sum("token_ids" in row for row in rows) == 35
    r = {row["case"]: row for row in rows}
    # Integer 1 is neither boolean true nor boolean false: the effort guard is
    # inactive, while the generation tail takes its ordinary open-think arm.
    assert r["numeric-thinking-source"]["text"].endswith(
        "<|im_start|>assistant\n<think>\n"
    )
    assert "Reasoning effort is set" not in r["numeric-thinking-source"]["text"]
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
    assert r["one-user"]["text"].endswith("<|im_start|>assistant\n<think>\n")
    if not source_override:
        assert "Reasoning effort is set to xhigh." in r["one-user"]["text"]
    assert "Reasoning effort is set" not in r["effort-medium"]["text"]
    assert "<parameter=yes>\ntrue\n</parameter>" in r["tool-arguments-types"]["text"]
    assert "<parameter=none>\nnull\n</parameter>" in r["tool-arguments-types"]["text"]
    assert "old thought" in r["assistant-history"]["text"]
    assert "old thought" not in r["preserve-false"]["text"]
