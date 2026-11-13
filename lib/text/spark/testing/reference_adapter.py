# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Original Spark Jinja oracle; native host normalization is explicit."""

import copy
import json

REVISION = "93d6aa90"
TEMPLATE_SHA256 = "93d6aa906ec72365efcab3c0109dba83148bdc9bec7c55848623d17483693eda"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    if case.get("native_arguments"):
        for message in request["messages"]:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(
                    call["function"]["arguments"]
                )
    kwargs = {"add_generation_prompt": True}
    kwargs.update(request.get("model_options", {}))
    if "enable_thinking" in request:
        kwargs["enable_thinking"] = request["enable_thinking"]
    if case.get("omit_generation_prompt"):
        kwargs.pop("add_generation_prompt", None)
    return request["messages"], request.get("tools"), kwargs


def error_record(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def check_reference(rows, source_override: bool):
    cases = {row["case"]: row for row in rows}
    if source_override:
        return
    assert cases["thinking-off"]["text"] == (
        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant."
        "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Hello"
        "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>"
    )
    assert "thinking_effort=high" in cases["one-user"]["text"]
    assert "thinking_effort=low" in cases["effort-low"]["text"]
    assert cases["missing-generation"]["text"] == cases["no-generation"]["text"]
    assert "<|Bot|><think> kept </think>A" in cases["assistant-history"]["text"]
    assert (
        "<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        in cases["glm-call"]["text"]
    )
    assert (
        "<|Tool|><tool_response>one</tool_response><tool_response>two</tool_response>"
        in cases["grouped-tools"]["text"]
    )
