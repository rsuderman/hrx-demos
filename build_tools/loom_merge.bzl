"""Native aggregation for Loom modules whose configuration is supplied later."""

def loom_merge(name, srcs, output_format = "bc", **kwargs):
    """Merge an exact source/filegroup closure without selecting live roots.

    The caller owns membership. Do not supply a merged product alongside its
    members. Final consumers must strict-link this input with real configuration.
    Text output is useful for exercising the same member set in source fixtures.
    """
    if output_format not in ["bc", "text"]:
        fail("loom_merge output_format must be bc or text")
    if len(srcs) != len({str(src): True for src in srcs}):
        fail("loom_merge contains duplicate source labels")
    linker = "@hrx_system//loom/src/loom/tools/loom-link"
    native.genrule(
        name = name,
        srcs = srcs,
        tools = [linker],
        outs = [name + (".loombc" if output_format == "bc" else ".loom")],
        cmd = "$(execpath %s) $(SRCS) --mode=merge --to=%s --output=$@" % (linker, output_format),
        **kwargs
    )
