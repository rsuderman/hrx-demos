# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local independent template inputs; never imports a production recognizer."""

import copy
import json

REVISION = "cd8e9439"
TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"


def render_kwargs(case, source_override: bool):
    request = copy.deepcopy(case["request"])
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
    assert len(rows) == 28
    assert sum("error" in row for row in rows) == 2
    assert sum("token_ids" in row for row in rows) == 26
