# Native GoogleTest tests

Run the basic Loomify framework test from the repository root:

```sh
bazel test //tools/loomify-gguf/testing:framework_test
bazel test //tools/loomify-gguf/testing:framework_test \
  --test_filter=LoomifyFrameworkTest.EmitsModelEntry
```

The root module declares a development-only GoogleTest dependency at the same
version as pinned HRX. HRX's own development dependencies are not inherited by
downstream modules, so tests depend directly on `@googletest//:gtest` and
`@googletest//:gtest_main`.
Include `gtest/gtest.h` and write ordinary `TEST`, `ASSERT_*` and
`EXPECT_*` assertions; no local test runner or additional dependency is needed.

For subsequent native tests, follow the adjacent `BUILD.bazel`: load the
execution-aware `cc_test` rule from
`@hrx_system//build_tools/bazel:cc_test.bzl`, set C++17 compiler options and
`linkstatic = True`, and add the library under test to `deps`. At the pinned
HRX version, the first-party test wrapper applies repository-relative link
configuration labels, so this external client uses its underlying rule directly.
Declare future file fixtures in `data` and resolve them through Bazel runfiles.

This smoke test exercises the existing source builder. Bytecode fixtures,
loading and registry behavior will have their own tests as they are added.
