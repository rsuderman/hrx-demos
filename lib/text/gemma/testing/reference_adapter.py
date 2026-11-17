# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local independent template inputs; never imports a production recognizer."""

import copy
import json

REVISION = "7de1c58e"
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
    kwargs = {"add_generation_prompt": True}
    kwargs.update(request.get("model_options", {}))
    if "enable_thinking" in request:
        kwargs["enable_thinking"] = request["enable_thinking"]
    if case.get("omit_generation_prompt"):
        kwargs.pop("add_generation_prompt", None)
    tools = request.get("tools")
    kwargs["bos_token"] = "<bos>"
    return (request["messages"], tools, kwargs)


def error_record(error):
    return {"error": type(error).__name__ + ": " + str(error)}


def check_reference(rows, source_override: bool):
    assert len(rows) == 33
    assert sum("error" in row for row in rows) == 8
    assert sum("token_ids" in row for row in rows) == 25
    by_name = {row["case"]: row for row in rows}
    if not source_override:
        user = "<bos><start_of_turn>user\nU<end_of_turn>\n"
        generation = "<start_of_turn>model\n"
        # The source tests user/non-user parity, not an assistant-only role
        # whitelist at odd ordinals. The native role field is a STRING.
        assert by_name["source-tool-history"]["text"] == (
            user + "<start_of_turn>tool\nT<end_of_turn>\n" + generation
        )
