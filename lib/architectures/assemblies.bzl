# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Finite model assembly membership; semantic readers and Loom sequences stay authored."""

load("//build_tools:model_assembly.bzl", "custom_assembly", "ordinary_assembly", "validate_assemblies")

ASSEMBLIES = [
    ordinary_assembly(
        family = "lfm2",
        resource_stem = "Lfm2",
        reader_symbol = "kLfm2Architecture",
        emitter_symbol = "EmitLfm2",
        publication = "text_and_turn",
        context_limit = 128000,
        required_operations = [],
        unsupported = "LFM2 requires ordinary greedy stages",
        architecture_srcs = [
            "//lib/architectures/lfm2:model.loom",
            "//lib/programs/lfm2:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/selection:greedy_argmax.loom",
            "//lib/kernels/epoch:count.loom",
            "//lib/kernels/publication:two_stops.loom",
        ],
    ),
    ordinary_assembly(
        family = "spark2_5",
        resource_stem = "Spark25",
        reader_symbol = "kSpark25Architecture",
        emitter_symbol = "EmitSpark25",
        publication = "eos",
        context_limit = 262144,
        required_operations = [],
        unsupported = "Spark2.5 requires ordinary greedy stages",
        architecture_srcs = [
            "//lib/architectures/spark2_5:model.loom",
            "//lib/programs/spark2_5:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/selection:greedy_argmax.loom",
            "//lib/kernels/epoch:count.loom",
            "//lib/kernels/publication:two_stops.loom",
        ],
    ),
    ordinary_assembly(
        family = "lfm3",
        resource_stem = "Lfm3",
        reader_symbol = "kLfm3Architecture",
        emitter_symbol = "EmitLfm3",
        publication = "text_and_turn",
        context_limit = 262144,
        required_operations = [],
        unsupported = "LFM3 requires ordinary greedy stages",
        architecture_srcs = [
            "//lib/architectures/lfm3:model.loom",
            "//lib/programs/lfm3:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/selection:greedy_argmax.loom",
            "//lib/kernels/epoch:count.loom",
            "//lib/kernels/publication:two_stops.loom",
        ],
    ),
    ordinary_assembly(
        family = "qwen2",
        resource_stem = "Qwen2",
        reader_symbol = "kQwen2Architecture",
        emitter_symbol = "EmitQwen2",
        publication = "text_and_turn",
        context_limit = 262144,
        required_operations = ["kScaledF16ProjectionV1"],
        unsupported = "Qwen2 requires ordinary greedy stages and matching generated capacities",
        architecture_srcs = [
            "//lib/architectures/qwen2:model.loom",
            "//lib/programs/qwen2:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/decoder:implementation_sources",
            "//lib/programs/projection:matrix_sources",
            "//lib/kernels/selection:greedy_argmax.loom",
            "//lib/kernels/epoch:count.loom",
            "//lib/kernels/publication:two_stops.loom",
        ],
    ),
    ordinary_assembly(
        family = "qwen3",
        resource_stem = "Qwen3",
        reader_symbol = "kQwen3Architecture",
        emitter_symbol = "EmitQwen3",
        publication = "text_and_turn",
        context_limit = 262144,
        required_operations = ["kScaledF16ProjectionV1"],
        unsupported = "Qwen3 requires ordinary greedy stages and matching generated capacities",
        architecture_srcs = [
            "//lib/architectures/qwen3:model.loom",
            "//lib/programs/qwen3:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/decoder:implementation_sources",
            "//lib/programs/projection:matrix_sources",
            "//lib/kernels/selection:greedy_argmax.loom",
            "//lib/kernels/epoch:count.loom",
            "//lib/kernels/publication:two_stops.loom",
        ],
    ),
    ordinary_assembly(
        family = "llama",
        resource_stem = "Llama",
        reader_symbol = "kLlamaArchitecture",
        emitter_symbol = "EmitLlama",
        publication = "three_stops",
        context_limit = 131072,
        required_operations = [],
        unsupported = "Llama 3B requires target-only greedy stages, empty model options and valid bounded capacities",
        publisher_symbol = "llama32_publish_epoch_id",
        architecture_srcs = [
            "//lib/architectures/llama:model.loom",
            "//lib/programs/llama:implementation_sources",
            "//lib/programs/ffn:interface.loom",
            "//lib/programs/ffn:ffn.loom",
        ],
        implementation_srcs = [
            "//lib/kernels/llama:configured_provider_sources",
        ],
    ),
    ordinary_assembly(
        family = "gemma3",
        resource_stem = "Gemma",
        reader_symbol = "kGemma3Architecture",
        emitter_symbol = "EmitGemma",
        publication = "eos_and_turn",
        context_limit = 262144,
        required_operations = [],
        unsupported = "Gemma3 requires ordinary greedy stages and matching generated capacities",
        architecture_srcs = [
            "//lib/architectures/gemma3:model.loom",
            "//lib/programs/gemma3:implementation_sources",
        ],
        implementation_srcs = [
            "//lib/kernels/gemma3:implementation_sources",
        ],
    ),
    custom_assembly(
        family = "qwen35",
        resource_stem = "Qwen",
        reader_symbol = "kQwen35Architecture",
        emitter_symbol = "EmitQwen",
        publication = "host_managed",
        context_limit = 262144,
        required_operations = ["kAttentionPrefillH256V1"],
        roots = ["generated.qwen_epoch"],
        products = [
            "//lib/architectures/qwen35:implementation",
            "//lib/architectures/qwen35:prepare_library",
            "//lib/architectures/qwen35:execution",
        ],
    ),
]

validate_assemblies(ASSEMBLIES)

def assembly(family):
    """Select a declared build entry; unknown architecture is a build error."""
    for entry in ASSEMBLIES:
        if entry.family == family:
            return entry
    fail("unknown declared assembly: " + family)
