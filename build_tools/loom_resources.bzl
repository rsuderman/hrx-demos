"""Private C++ paths from the same declared Loom products used as runfiles."""

def _loom_resources_impl(ctx):
    if not ctx.attr.products:
        fail("Loom resources require declared products")
    entries = []
    identifiers = {}
    for target, identifier in ctx.attr.products.items():
        if not identifier or identifier[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
            fail("Loom resource identifier must be a C++ identifier: " + identifier)
        for char in identifier.elems():
            if char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789":
                fail("Loom resource identifier must be a C++ identifier: " + identifier)
        if identifier == "kAllProducts":
            fail("reserved Loom resource identifier: " + identifier)
        if identifier in identifiers:
            fail("duplicate Loom resource identifier: " + identifier)
        identifiers[identifier] = True
        files = target[DefaultInfo].files.to_list()
        if len(files) != 1 or files[0].is_directory:
            fail("Loom resource must declare exactly one file: " + str(target.label))
        path = files[0].short_path
        if path.startswith("/") or "\\" in path or ":" in path or any([part in ["", ".", ".."] for part in path.split("/")]):
            fail("Loom resource must have a main-repository relative path: " + path)
        entries.append((identifier, path))
    lines = [
        "// Generated from declared Loom resource outputs; do not edit.",
        "#ifndef HRX_SOLUTIONS_LOOMIFY_LIBRARY_RESOURCES_H_",
        "#define HRX_SOLUTIONS_LOOMIFY_LIBRARY_RESOURCES_H_",
        "namespace hrx::loomify_gguf::resources {",
    ]
    for identifier, path in sorted(entries):
        escaped = path.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        lines.append('inline constexpr const char %s[] = "%s";' % (identifier, escaped))
    lines.append("inline constexpr const char* kAllProducts[] = {")
    for identifier, _ in sorted(entries):
        lines.append("    " + identifier + ",")
    lines.extend([
        "};",
        "}  // namespace hrx::loomify_gguf::resources",
        "#endif  // HRX_SOLUTIONS_LOOMIFY_LIBRARY_RESOURCES_H_",
        "",
    ])
    ctx.actions.write(ctx.outputs.out, "\n".join(lines))
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

loom_resources = rule(
    implementation = _loom_resources_impl,
    attrs = {
        "products": attr.label_keyed_string_dict(allow_files = True, mandatory = True),
        "out": attr.output(mandatory = True),
    },
)
