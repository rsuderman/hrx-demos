# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Independent anchors on the pinned oracle; not Loom implementation proof."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("reference", type=Path)
a = p.parse_args()
rows = json.loads(a.reference.read_text())
r = {(row["revision"], row["case"]): row for row in rows}
assert len(rows) == 82 and sum("error" in row for row in rows) == 13
assert sum("token_ids" in row for row in rows) == 69
assert r["7f0e5290", "one-user"]["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
assert r["c3cf9e34", "one-user"]["text"].endswith("<|im_start|>assistant\n<think>\n")
assert "Reasoning effort is set to xhigh." in r["c3cf9e34", "one-user"]["text"]
assert "Reasoning effort is set" not in r["c3cf9e34", "effort-medium"]["text"]
for revision in ("7f0e5290", "c3cf9e34"):
    assert r[revision, "tool-arguments-float-exponent"]["text"] == r[revision, "tool-arguments-float-decimal"]["text"]
    assert "<parameter=n>\n1.0\n</parameter>" in r[revision, "tool-arguments-float-exponent"]["text"]
    assert "<parameter=n>\n-0.0\n</parameter>" in r[revision, "tool-arguments-negative-zero"]["text"]
    assert "café <>&\n" in r[revision, "tool-arguments-types"]["text"]
    assert "<parameter=n>\n1\n</parameter>" in r[revision, "tool-arguments-types-integer"]["text"]
assert "<parameter=yes>\nTrue\n</parameter>" in r["7f0e5290", "tool-arguments-types"]["text"]
assert "<parameter=none>\nNone\n</parameter>" in r["7f0e5290", "tool-arguments-types"]["text"]
assert "<parameter=yes>\ntrue\n</parameter>" in r["c3cf9e34", "tool-arguments-types"]["text"]
assert "<parameter=none>\nnull\n</parameter>" in r["c3cf9e34", "tool-arguments-types"]["text"]
assert "old thought" not in r["7f0e5290", "assistant-history"]["text"]
assert "new thought" in r["7f0e5290", "assistant-history"]["text"]
assert "old thought" in r["c3cf9e34", "assistant-history"]["text"]
assert "old thought" not in r["c3cf9e34", "preserve-false"]["text"]
for (revision, _), row in r.items():
    if revision == "5816fce1":
        assert row["text"].count("<|begin_of_text|>") == 1
        assert "Today Date: 26 Jul 2024\n\n" in row["text"]
        assert row["token_ids"][0] == 128000 and row["token_ids"].count(128000) == 1
print("PASS: 82 independent oracle rows, 69 complete token vectors, defaults/history/scalar/date/BOS anchors")
