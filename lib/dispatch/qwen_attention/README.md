# Qwen 0.8B complete attention layer

`QwenAttentionRegistrations()` supplies the additive native command family at
`//lib/dispatch/qwen_attention:builder`. It consumes the model-owned complete
`QwenAttentionLayer` operation and the common `QwenEpoch` session binding. The
returned value is the existing residual; K/V remain explicit mutable effects.

The supported profile has E=1024, F=3584, Hq=8, Hkv=2, D=256, rotary width 64,
fixed root/schedule T<=512, and one ordinary epoch span with runtime N<=T.
Logical caches use C; native page placement uses rounded P. All eleven named
weights require their canonical original-layer keys and one scoped bank.
Logical K/V identities must match the configured epoch roles, original layer,
attention ordinal and common layer slot. Geometry, loaded encoding selectors,
bank keys, native ownership and command
ABI are checked before storage/constants are staged. Native dependent-value
validation still applies to the final checked call; discard owners after a
staging failure.

The numerical closure includes both learned layer norms, Q/K/V preparation,
actual gfx11 WMMA attention, output projection/residual and the shared whole FFN
with its residual. Private attended output is cleared for all T rows before
active-span writes. Initialized transient padding rows may change through the
dense children. Persistent inactive cache cells, other planes and bytes outside
owned footprints must remain unchanged.

`testing:builder_test` proves same-owner recovery for every incorrect role key
and mixed bank, boundary aliases, normal session emission, residual/state
mappings, state-association mismatch recovery, strict linking, both serialization
formats and Q5/Q6 artifact builds. The manual `testing:real_gguf_test` enumerates
all 66 dense-layer roles from the immutable default Q5 GGUF header and compiles
those six real descriptor calls without reading checkpoint payloads.
The numerical primitive tests are in `//lib/kernels/qwen/attention/testing`.
`testing/run_layer_fixture.sh` runs the complete-layer numerical proof using
fresh serialized test outputs, native command-product reflection, and the
bounded `layer_fixture.py` scalar oracle. The continuation fixture starts with
initialized history and snapshots each dependent complete call across positions
63/64, a later chunk, a length-one continuation and reset to position zero; it
uses nonzero packed starts and exact persistent-history/other-plane guards.
Its explicit inputs are a standalone
Python interpreter, a native reflection reader, the native checker, the emitted
module, Q/K encoding 5 or 6, and an external output directory. Reflection uses
an allowed native test package because that reader is currently private to
HRX; no runtime or visibility extension is part of this family.
