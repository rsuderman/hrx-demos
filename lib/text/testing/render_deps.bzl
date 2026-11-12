# Exact native dependencies of the test-only external SDK consumer.
# chat/input/tool_xml are SDK-private. This list does not make them public.
RENDER_DEPS = [
    "@hrx_system//loom/binding/c:loomc",
    "@hrx_system//loom/binding/c/target/vm",
    "@hrx_system//loom/src/loom/runtime/program:input",
    "@hrx_system//loom/src/loom/runtime/program:json",
    "@hrx_system//loom/src/loom/runtime/program:program",
    "@hrx_system//loom/src/loom/serve/service/text:chat",
    "@hrx_system//loom/src/loom/serve/service/text:input",
    "@hrx_system//loom/src/loom/serve/service/text:tool_xml",
    "@hrx_system//runtime/src/iree/base",
    "@hrx_system//runtime/src/iree/base/internal:json",
    "@hrx_system//runtime/src/iree/base/internal:unicode",
    "@hrx_system//runtime/src/iree/tokenizer:tokenizer",
    "@hrx_system//runtime/src/iree/tokenizer/format/huggingface:tokenizer_json",
    "@hrx_system//runtime/src/iree/vm",
    "@googletest//:gtest",
    "@googletest//:gtest_main",
]
