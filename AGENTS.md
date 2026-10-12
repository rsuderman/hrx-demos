# HRX Demos Agent Guide

This repository builds and publishes the `hrx-demos` Python wheel for the Loom
Serve text server from the HRX commit pinned in `deps.lock`.

Keep this repository focused on the wheel, its thin Python launcher, pinned
build inputs, and GitHub Actions workflows. Model kernels, runtime code,
checkpoints, and model-specific tests belong to HRX or external artifact roots.
The wheel build stages only the selected model's source catalog and required
runtime libraries.

Do not commit downloaded model weights, generated serving outputs, or machine-
specific paths. Do not change the HRX source pin without updating `deps.lock`
and reviewing the resulting wheel build.
