# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Known compact ChatML inputs; independent of the production selector."""

import copy
import json

REVISION = "f45005c5"
TEMPLATE_SHA256 = "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    if case.get("native_arguments"):
        # The existing Serve interface transports tool arguments as JSON
        # strings. Supply its decoded mapping to the original template, as in
        # the existing explicit-reasoning adapter; keep raw-template error
        # cases separate and never change their inputs.
        for message in request["messages"]:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(
                    call["function"]["arguments"]
                )
    # Original GGUF declares this BOS piece; its template inserts it explicitly.
    # Unmarked raw-template cases retain their original argument representation.
    kwargs = {"bos_token": "<|startoftext|>", "add_generation_prompt": True}
    kwargs.update(request.get("model_options", {}))
    if "enable_thinking" in request:
        kwargs["enable_thinking"] = request["enable_thinking"]
    if case.get("omit_generation_prompt"):
        kwargs.pop("add_generation_prompt", None)
    return request["messages"], request.get("tools"), kwargs


def error_record(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def check_reference(rows, source_override: bool):
    assert len(rows) == 41
    assert sum("error" in row for row in rows) == 1
    assert sum("native_admission_error" in row for row in rows) == 7
    r = {row["case"]: row for row in rows}
    assert r["one-user"]["text"] == (
        "<|startoftext|><|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    assert r["thinking-off"]["text"].endswith("<think></think>")
    assert r["no-generation"]["text"] == r["missing-generation"]["text"]
    assert "<think>first</think>Answer" in r["reasoning-precedence"]["text"]
    assert "<think></think>Answer" in r["old-inline-reasoning"]["text"]
    assert "<think>kept</think>" in r["preserve-thinking"]["text"]
    assert "<think>kept</think>Answer" in r["assistant-without-user"]["text"]
    assert "<think>kept</think>Answer" in r["system-assistant-without-user"]["text"]
    assert '<tools>\n{"name": "f"' in r["tools-wrapped-schema"]["text"]
    assert "<parameter=yes>\ntrue\n</parameter>" in r["xml-call-values"]["text"]
    assert "</tool_call>\n\n<tool_call>" in r["xml-multiple-calls"]["text"]
    assert "<|im_start|>tool\n<tool_response>\none" in r["tool-responses"]["text"]
    assert "<additional_tools>" in r["tool-reference"]["text"]
    assert "part1\npart2" in r["merge-tool-parts"]["text"]
    assert "<tool_response>\none\ntwo\n</tool_response>" in r["tool-text-parts"]["text"]
