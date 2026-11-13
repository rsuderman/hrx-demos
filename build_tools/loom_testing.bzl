# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Paired representations of the same explicit provider/check boundary."""

load("@hrx_system//loom/build_tools/bazel:defs.bzl", "loom_test")
load("//build_tools:loom_merge.bzl", "loom_merge")

def loom_checks(name, provider_name, checks, execution_profiles, **kwargs):
    """Add checks sharing an existing text/BC provider pair."""
    for output_format in ["text", "bc"]:
        loom_test(
            name = name + "_" + output_format + "_test",
            srcs = checks + [":" + provider_name + "_" + output_format],
            benchmark_smoke = False,
            execution_profiles = execution_profiles,
            **kwargs
        )

def loom_provider_checks(name, provider_name, provider_srcs, checks, execution_profiles, **kwargs):
    """Merge providers separately from checks in both maintained representations."""
    for output_format in ["text", "bc"]:
        loom_merge(
            name = provider_name + "_" + output_format,
            testonly = True,
            srcs = provider_srcs,
            output_format = output_format,
        )
    loom_checks(name, provider_name, checks, execution_profiles, **kwargs)
