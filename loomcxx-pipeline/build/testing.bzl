"""Shared static-fixture matrix and existing AMDGPU test launcher."""

load(":loomcxx.bzl", "loom_fixture")
load("@hrx_system//loom/build_tools/bazel:defs.bzl", "loom_test")
load("@hrx_system//loom/target/amdgpu:execution_profiles.bzl", "AMDGPU_HARDWARE_PROFILE")

def loom_gpu_test(name, srcs, deps, configs):
    """Runs a native numerical/contract fixture with the common target policy."""
    loom_test(
        name = name,
        srcs = srcs,
        deps = deps,
        configs = configs,
        benchmark_smoke = False,
        execution_profiles = [AMDGPU_HARDWARE_PROFILE],
    )

def loom_test_matrix(name, fixture, definitions, cases):
    """Expands family-owned cases against each explicitly selected definition set.

    cases contain name, substitutions and configs; definitions maps authoring
    names to dependency labels. Neither numerical arithmetic nor model policy
    belongs here.
    """
    for case in cases:
        check = name + "_checks_" + case["name"]
        loom_fixture(name = check, src = fixture, substitutions = case["substitutions"])
        for authoring, deps in definitions.items():
            loom_gpu_test(
                name = name + "_" + authoring + "_" + case["name"],
                srcs = [":" + check],
                deps = deps,
                configs = case["configs"],
            )
    for authoring in definitions:
        native.test_suite(
            name = name + "_" + authoring + "_execution_test",
            tests = [":" + name + "_" + authoring + "_" + case["name"] for case in cases],
        )
