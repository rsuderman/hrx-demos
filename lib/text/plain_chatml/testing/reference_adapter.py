# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent plain ChatML inputs, with explicit raw-template default proof."""

import copy

REVISION = "58c1a1f0"
TEMPLATE_SHA256 = "58c1a1f04baa7adaeaba1f90267c3d57d9396f0c1f0129bdde9e76d3c6784af7"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    # This source only concatenates content as text. Apply the agreed native
    # text-only projection independently; never alter the recorded wire request.
    for message in request["messages"]:
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif isinstance(content, list):
            assert all(part["type"] == "text" for part in content)
            message["content"] = "".join(part["text"] for part in content)
    kwargs = {"add_generation_prompt": True}
    kwargs.update(request.get("model_options", {}))
    if case.get("omit_generation_prompt"):
        kwargs.pop("add_generation_prompt", None)
    return request["messages"], request.get("tools"), kwargs


def error_record(error):
    return {"error_type": type(error).__name__, "error": str(error)}


def check_reference(rows, source_override: bool):
    assert len(rows) == 28
    assert sum("error" in row for row in rows) == 0
    assert sum("native_admission_error" in row for row in rows) == 0
    assert sum("token_ids" in row for row in rows) == 28
    by_name = {row["case"]: row for row in rows}
    assert by_name["null-content"]["text"] == by_name["missing-content"]["text"]
    assert by_name["array-content"]["text"] == by_name["one-user"]["text"]
    # Raw template omission is false. Its paired native request explicitly
    # supplies false, rather than confusing this with Serve's default true.
    omitted = by_name["raw-default-false"]
    assert "add_generation_prompt" not in omitted["kwargs"]
    assert omitted["text"] == by_name["generation-false"]["text"]
    if not source_override:
        assert by_name["one-user"]["text"] == (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        )
