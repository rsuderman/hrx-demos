# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Finite build-owned assembly joins; no numerical or text semantics."""

load("//build_tools:loom_merge.bzl", "loom_merge")

_PUBLICATIONS = {
    "eos": "kEos",
    "text_and_turn": "kTextAndTurn",
    "eos_and_turn": "kEosAndTurn",
    "three_stops": "kThreeStops",
    "host_managed": "kHostManaged",
}

def _identifier(value):
    if not value or value[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
        fail("invalid assembly identifier: " + value)
    for c in value.elems():
        if c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789":
            fail("invalid assembly identifier: " + value)

def _label(value):
    if not value.startswith("//") or ":" not in value or "\\" in value:
        fail("invalid assembly source/product label: " + value)
    for c in value.elems():
        if c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789/.:+-":
            fail("invalid assembly source/product label: " + value)
    for part in value.replace(":", "/").split("/")[2:]:
        if part in ["", ".", ".."]:
            fail("invalid assembly source/product path: " + value)

def ordinary_assembly(family, reader_symbol, emitter_symbol, resource_stem, architecture_srcs, implementation_srcs, publication, context_limit, required_operations, unsupported, publisher_symbol = None):
    """Declare an existing ordinary seven-buffer boundary and its source closure."""
    return struct(
        family = family,
        reader_symbol = reader_symbol,
        emitter_symbol = emitter_symbol,
        resource_stem = resource_stem,
        architecture_srcs = architecture_srcs,
        implementation_srcs = implementation_srcs,
        publication = publication,
        context_limit = context_limit,
        required_operations = required_operations,
        unsupported = unsupported,
        publisher_symbol = publisher_symbol or family + "_publish_epoch_id",
        roots = ["generated." + family + "_epoch"],
        products = ["//lib/architectures/" + family + ":" + name for name in ["implementation", "prepare_library", "execution"]],
        ordinary = True,
    )

def custom_assembly(family, reader_symbol, emitter_symbol, resource_stem, publication, context_limit, required_operations, roots, products):
    """Register real custom products without generating their host/numerical ABI."""
    return struct(
        family = family,
        reader_symbol = reader_symbol,
        emitter_symbol = emitter_symbol,
        resource_stem = resource_stem,
        publication = publication,
        context_limit = context_limit,
        required_operations = required_operations,
        roots = roots,
        products = products,
        ordinary = False,
    )

def validate_assemblies(entries):
    """Reject ambiguous finite membership before producing any generated source."""
    families = {}
    symbols = {}
    products = {}
    stems = {}
    for entry in entries:
        for value in [entry.family, entry.reader_symbol, entry.emitter_symbol, entry.resource_stem] + entry.required_operations:
            _identifier(value)
        if entry.family in families:
            fail("duplicate assembly family: " + entry.family)
        families[entry.family] = True
        for symbol in [entry.reader_symbol, entry.emitter_symbol]:
            if symbol in symbols:
                fail("duplicate assembly symbol: " + symbol)
            symbols[symbol] = True
        if entry.resource_stem in stems:
            fail("duplicate assembly output identity: " + entry.resource_stem)
        stems[entry.resource_stem] = True
        if entry.publication not in _PUBLICATIONS or entry.context_limit < 1 or len(entry.products) != 3 or not entry.roots:
            fail("invalid assembly policy/products: " + entry.family)
        for product in entry.products:
            _label(product)
            if product in products:
                fail("duplicate assembly product: " + product)
            products[product] = True
        if entry.ordinary:
            _identifier(entry.publisher_symbol)
            if entry.publication == "host_managed":
                fail("host-managed assembly requires custom products")
            for src in entry.architecture_srcs + entry.implementation_srcs:
                _label(src)

def logical_sources(entries):
    return ["//tools/model:architectures/" + entry.family + ".cc" for entry in entries]

def native_sources(entries):
    return ["//lib/loomify:architectures/" + entry.family + ".cc" for entry in entries]

def assembly_products(entries):
    return {entry.products[i]: "k" + entry.resource_stem + suffix for entry in entries for i, suffix in enumerate(["Implementation", "Preparation", "Execution"])}

def _write_sources_impl(ctx):
    ctx.actions.write(ctx.outputs.header, ctx.attr.header_content)
    ctx.actions.write(ctx.outputs.source, ctx.attr.source_content)
    return [DefaultInfo(files = depset([ctx.outputs.source]))]

_write_sources = rule(
    implementation = _write_sources_impl,
    attrs = {
        "header_content": attr.string(mandatory = True),
        "source_content": attr.string(mandatory = True),
        "header": attr.output(mandatory = True),
        "source": attr.output(mandatory = True),
    },
)

_BANNER = "// Generated from finite assembly declarations; do not edit.\n"

def logical_registry(name, entries):
    """Project logical membership without resolving any native product labels."""
    validate_assemblies(entries)
    header = _BANNER + '#pragma once\n#include "tools/model/architectures.h"\nnamespace hrx::model::internal {\n'
    header += "\n".join(["extern const ModelArchitecture " + e.reader_symbol + ";" for e in entries]) + "\n}\n"
    source = _BANNER + '#include "tools/model/assembly_registry.h"\nnamespace hrx::model::internal {\n'
    source += "const ModelArchitecture *SelectModelArchitecture(std::string_view family) {\n  static const ModelArchitecture *const entries[] = {\n"
    source += "\n".join(["    &" + e.reader_symbol + "," for e in entries])
    source += "\n  };\n  for (const auto *entry : entries)\n    if (family == entry->family) return entry;\n  return nullptr;\n}\n}\n"
    _write_sources(name = name, header = "assembly_registry.h", source = "assembly_registry.cc", header_content = header, source_content = source)

def native_registry(name, entries):
    """Emit the existing ArchitectureFiles selector; paths stay resource-owned."""
    validate_assemblies(entries)
    header = _BANNER + '#pragma once\n#include "lib/loomify/configuration.h"\nnamespace hrx::loomify_gguf::internal {\n'
    signature = "(const model::ModelDescriptor &, const std::string &, const ConfigurationSettings &, ConfigurationFacts *, std::string *);"
    header += "\n".join(["bool " + e.emitter_symbol + signature for e in entries]) + "\n}\n"
    source = _BANNER + '#include "lib/loomify/architectures.h"\n#include "lib/loomify/assembly_registry.h"\n#include "lib/serving/library_resources.h"\nnamespace hrx::loomify_gguf {\n'
    source += "const ArchitectureFiles *SelectArchitecture(std::string_view family) {\n  static const ArchitectureFiles entries[] = {\n"
    for e in entries:
        resources = ["resources::k" + e.resource_stem + suffix for suffix in ["Implementation", "Preparation", "Execution"]]
        source += "    {" + ", ".join([json.encode(e.family)] + resources + ["resources::kWeights", json.encode("loomify-" + e.family), "{" + ", ".join([json.encode(root) for root in e.roots]) + "}", "{" + ", ".join(["RequiredOperation::" + op for op in e.required_operations]) + "}", "internal::" + e.emitter_symbol, "PublicationLayout::" + _PUBLICATIONS[e.publication], str(e.context_limit)]) + "},\n"
    source += "  };\n  for (const auto &entry : entries)\n    if (family == entry.family) return &entry;\n  return nullptr;\n}\n}\n"
    _write_sources(name = name, header = "assembly_registry.h", source = "assembly_registry.cc", header_content = header, source_content = source)

def _write_loom_impl(ctx):
    ctx.actions.write(ctx.outputs.out, ctx.attr.content)
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_write_loom = rule(
    implementation = _write_loom_impl,
    attrs = {"content": attr.string(mandatory = True), "out": attr.output(mandatory = True)},
)

_PRINTABLE = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
_HEX = "0123456789abcdef"

def _hex_ascii(value):
    result = ""
    for c in value.elems():
        number = _PRINTABLE.find(c) + 32
        if number < 32:
            fail("ordinary diagnostic/root must use printable ASCII")
        result += _HEX[number // 16] + _HEX[number % 16]
    return result

def _prepare(entry):
    return _BANNER + '''global.rodata.def @ordinary_root = bytes("ROOT_HEX")
global.rodata.def @ordinary_diagnostic = bytes("DIAGNOSTIC_HEX")
func.def public inline @ordinary.epoch_root() -> (buffer) {
  %value = global.load @ordinary_root : buffer
  func.return %value : buffer
}
func.def public inline @ordinary.unsupported() -> (buffer) {
  %value = global.load @ordinary_diagnostic : buffer
  func.return %value : buffer
}
'''.replace("ROOT_HEX", _hex_ascii(entry.roots[0])).replace("DIAGNOSTIC_HEX", _hex_ascii(entry.unsupported))

def _epoch(entry):
    return _BANNER + """command.decl @loomify.FAMILY.model() launch(%parameters: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %state: buffer, %input_ids: buffer, %selected_ids: buffer)
kernel.decl @PUBLISHER() launch(%epoch: buffer, %selected_ids: buffer, %output_ids: buffer)
command.def public @generated.FAMILY_epoch() launch(%parameters: buffer, %residual: buffer, %epoch: buffer, %rows: buffer, %state: buffer, %input_ids: buffer, %output_ids: buffer) {
  %selected_bytes = index.constant 16 : offset
  %selected_ids = buffer.alloca<global> align(256) %selected_bytes : buffer
  command.launch @loomify.FAMILY.model(%parameters, %residual, %epoch, %rows, %state, %input_ids, %selected_ids) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.launch @PUBLISHER(%epoch, %selected_ids, %output_ids) : (buffer, buffer, buffer)
  command.return
}
""".replace("FAMILY", entry.family).replace("PUBLISHER", entry.publisher_symbol)

def _control(entry):
    slots = {
        "eos": ["eos_id"],
        "text_and_turn": ["end_text_id", "end_turn_id"],
        "eos_and_turn": ["eos_id", "end_turn_id"],
        "three_stops": ["end_text_id", "end_message_id", "end_turn_id"],
    }[entry.publication]
    source = _BANNER
    prefix = entry.family + ".publication."
    for slot in slots:
        source += "config.decl @" + prefix + slot + " : i32\n"
    source += "func.def " + ("inline " if entry.publication == "three_stops" else "") + "@loomify.serving.terminal_id() -> (i32) {\n  %terminal = config.get @" + prefix + slots[-1] + " : i32\n  func.return %terminal : i32\n}\n"
    if entry.publication == "three_stops":
        source += """func.def public inline @loomify_serving_native_token(%token: i32, %terminal: i32) -> (i32) {
  %first = config.get @PREFIXend_text_id : i32
  %second = config.get @PREFIXend_message_id : i32
  %third = config.get @PREFIXend_turn_id : i32
  %a = scalar.cmpi eq, %token, %first : i32
  %b = scalar.cmpi eq, %token, %second : i32
  %c = scalar.cmpi eq, %token, %third : i32
  %ab = scalar.ori %a, %b : i1
  %stop = scalar.ori %ab, %c : i1
  %mapped = scf.select %stop, %terminal, %token : i32
  func.return %mapped : i32
}
""".replace("PREFIX", prefix)
    else:
        count = len(slots)
        helper = "llm.publication.map_" + ("one_stop" if count == 1 else "two_stops")
        pure = "pure " if entry.publication == "eos_and_turn" else ""
        signature = "%token: i32, " + ", ".join(["%stop" + str(i) + ": i32" for i in range(count)]) + ", %terminal: i32"
        source += "func.decl " + pure + "@" + helper + "(" + signature + ") -> (i32)\n"
        source += "func.def public inline @loomify_serving_native_token(%token: i32, %terminal: i32) -> (i32) {\n"
        for i, slot in enumerate(slots):
            source += "  %stop" + str(i) + " = config.get @" + prefix + slot + " : i32\n"
        source += "  %mapped = func.call " + pure + "@" + helper + "(%token, " + ", ".join(["%stop" + str(i) for i in range(count)]) + ", %terminal) : (" + ", ".join(["i32"] * (count + 2)) + ") -> (i32)\n  func.return %mapped : i32\n}\n"
    return source

def ordinary_products(entry):
    """Generate only the existing ordinary shell; preserve authored model order."""
    if not entry.ordinary or native.package_name() != "lib/architectures/" + entry.family:
        fail("ordinary products must be declared in their owning architecture package")
    _write_loom(name = "ordinary_prepare_source", out = "generated_prepare.loom", content = _prepare(entry))
    _write_loom(name = "ordinary_epoch_source", out = "generated_epoch.loom", content = _epoch(entry))
    _write_loom(name = "ordinary_control_source", out = "generated_control_core.loom", content = _control(entry))
    architecture_srcs = entry.architecture_srcs + [":ordinary_epoch_source"]
    native.filegroup(name = "architecture_sources", srcs = architecture_srcs)
    srcs = architecture_srcs + entry.implementation_srcs
    loom_merge(name = "implementation", srcs = srcs)
    loom_merge(name = "implementation_text", srcs = srcs, testonly = True, output_format = "text")
    loom_merge(name = "prepare_library", srcs = [":ordinary_prepare_source", "//lib/serving:ordinary_prepare.loom"])
    control_srcs = [":ordinary_control_source", "//loom-pipeline:llm/common/fixed_epoch_abi.loom", "//lib/serving:ordinary_control.loom"]
    if entry.publication != "three_stops":
        control_srcs.append("//lib/kernels/publication:two_stops.loom")
    loom_merge(name = "execution", srcs = control_srcs)
