# Selected GGUF FFN compilation checkpoint

`hrx::dispatch::FfnRegion` selects a recorded normalized FFN from an existing
`LlamaStructure`, retains its owner and records typed dispatch IDs in the existing
operation order. It checks the selected layer's parameter roles and dataflow;
there is no second graph or native numerical recipe. Selection does not emit or
mutate the graph.

After loading the selected registrations into the native module, create a
`NativeEmissionSession` over that same immutable graph. Bind the existing residual
and, for prefill, the existing control buffer. `EmitOneRow(session, parameters,
error)` declares the owner's named parameters through `BindRegionParameters`,
asserts the proof entrypoint's `begin=0,count=1` specialization, and emits the
existing operations. Output and packing storage is generated internally. The
residual keeps its ReadWrite identity and native buffer.

This entrypoint is an explicit static one-row proof. Its prefill caller must
supply control count 1. It does not implement dynamic active-row consumer mapping
or a general Loom Serve entrypoint. Current provider admission limits widths to
3072/8192, gate/up to Q4_K, down to Q4_K or Q6_K and RMS epsilon to the selected
provider's F32 value. GGUF F32 epsilon facts are compared at that precision without
rewriting the facts. Discard the native owner after an emission failure.

`//lib/dispatch/testing:llama_region_test` checks selection and rejection behavior.
The manual `//lib/dispatch/testing:llama_region_gguf_test` requires an explicit
external `--test_env=LOOMIFY_TEST_GGUF=/path/to/the/known/Llama.gguf`. It reads the
owned header, selects a resolved FFN for both phases, checks three prefill/four
decode outer calls, named F32/I8 parameter views, output wiring and residual alias,
and roundtrips/compiles both source and bytecode with the actual registered
providers. Generated artifacts use the test's external output directory.

That test does not read weight payloads or execute the FFN. Compiled root placement,
GGUF payload preparation and independent numerical/intermediate/guard checks
remain separate proofs. A successful compile is not numerical qualification.
