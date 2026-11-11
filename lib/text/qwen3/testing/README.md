# Qwen3 template and transport fixtures

`render_cases.json` contains 40 independent-template inputs: the original 35
cases and five native transport counterparts. The shared
`../../qwen35/testing/render_reference.py` driver evaluates the actual selected
GGUF template with pinned HF/Jinja helpers and tokenizer JSON. Expected bytes
and token vectors are never captured from the Loom implementation.

Five original inputs belong to the broader template domain, but are rejected
by the current typed native chat API. Their `native_admission_error` records
the required negative diagnostic; `native_counterpart` names a separate valid
transport case. The original requests and template expectations are retained.

The `-native` counterparts use `type: function`, JSON-string tool arguments,
and an omitted optional reasoning field in place of explicit null. They test
the same intended rendered content through the supported public request
schema. String argument spacing and Unicode remain significant; the verbatim
case must not be parsed and reserialized. `template_case` links a counterpart
to its original template-domain input.

The SDK test in `../../qwen35/testing/render_test.cc` checks all 40 rows in both
SOURCE and bytecode: five exact host-admission rejections, one expected template
rejection, and 34 exact rendered-byte/raw-token/prepared-token comparisons.
The newline-token mismatch remains a failing parity assertion. Optional null
admission and exact floating-point tool JSON parity are separate work; these
fixtures do not broaden the runtime request schema or waive those gaps.
