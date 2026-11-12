"""Exact-pin C++ import and ordinary verified module products, shared by families."""

_HEADERS = [
    "@hrx_system//loom/src/loom/import/cxx:include/loomcxx/" + path
    for path in [
        "atomic.h",
        "buffer.h",
        "cache.h",
        "encoding_type.h",
        "index.h",
        "kernel.h",
        "predicate.h",
        "scalar.h",
        "shape.h",
        "target.h",
        "target/amdgpu.h",
        "view.h",
    ]
]

def loomcxx_module(name, src, bridges = [], products = ["full", "providers", "callers"]):
    """Imports one computation source and assembles explicitly mapped public ABIs.

    The source may guard providers with LOOMCXX_CALLERS_ONLY and kernel entries
    with LOOMCXX_PROVIDERS_ONLY. Products use name[_providers|_callers].loombc.
    """
    rows = []
    for family, provider, imported_family, mode, types in bridges:
        for symbol in [family, provider, imported_family]:
            if not symbol:
                fail("empty ABI symbol")
            for char in symbol.elems():
                if char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._":
                    fail("unsupported ABI symbol spelling")
        if mode not in ["raw", "zero-origins"]:
            fail("unsupported ABI adaptation: " + mode)
        if not types or (mode == "zero-origins" and "buffer" not in types):
            fail("ABI mapping requires argument types and zero-origins requires buffers")
        for arg_type in types:
            if arg_type not in ["index", "offset", "i1", "f32", "buffer"]:
                fail("unsupported ABI argument type: " + arg_type)
        rows.append(" ".join([family, provider, imported_family, mode, ",".join(types)]))
    abi = ":" + name + "_abi"
    if rows:
        native.genrule(
            name = name + "_abi",
            outs = [name + ".abi"],
            cmd = "printf '%s\\n' " + " ".join(["'" + row + "'" for row in rows]) + " > $@",
        )
    for product in products:
        if product not in ["full", "providers", "callers"]:
            fail("unsupported loomcxx product: " + product)
        target = name + ("_" + product if product != "full" else "")
        define = "--D=LOOMCXX_" + product.upper() + "_ONLY=1 " if product != "full" else ""
        native.genrule(
            name = target + "_import",
            srcs = [src] + _HEADERS,
            outs = [target + "_import.loombc"],
            cmd = "header=$(location @hrx_system//loom/src/loom/import/cxx:include/loomcxx/buffer.h); " +
                  "$(location @hrx_system//loom/src/loom/tools/loom-import-cxx) " +
                  "--I=$${header%loomcxx/buffer.h} " + define +
                  "--to=bc --output=$@ $(location " + src + ")",
            tools = ["@hrx_system//loom/src/loom/tools/loom-import-cxx"],
        )
        native.genrule(
            name = target,
            srcs = [":" + target + "_import"] + ([abi] if rows and product != "callers" else []),
            outs = [target + ".loombc"],
            cmd = "$(location //loomcxx-pipeline/support:assemble) " +
                  "$(location :" + target + "_import) $@" +
                  (" $(location " + abi + ")" if rows and product != "callers" else ""),
            tools = ["//loomcxx-pipeline/support:assemble"],
        )

def loom_fixture(name, src, substitutions):
    """Expands static fixture extents; numerical source stays generic.

    Keys and values must be simple literal tokens, not shell/sed expressions.
    """
    for key, value in substitutions.items():
        for char in key.elems():
            if char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@":
                fail("fixture keys must be literal identifier tokens")
        for char in str(value).elems():
            if char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-":
                fail("fixture values must be literal tokens")
    native.genrule(
        name = name,
        srcs = [src],
        outs = [name + ".loom"],
        cmd = "sed " + " ".join([
            "-e 's/" + key + "/" + str(value) + "/g'"
            for key, value in substitutions.items()
        ]) + " $< > $@",
    )
