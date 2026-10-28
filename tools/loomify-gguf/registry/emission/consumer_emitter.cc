// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/emission/consumer_emitter.h"

#include "loomc/source.h"

namespace hrx::loomify_gguf {
namespace {

constexpr char kScaleMinProbe[] = R"loom(
template.decl @ggml.qk.scale_min(%weight: buffer, %base: offset, %group: index) -> (f32, f32)

kernel.def export("registry_scale_min_probe") @registry_scale_min_probe() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%weight: buffer, %output: buffer) {
  %base = index.constant 0 : offset
  %group = index.constant 0 : index
  %scale, %minimum = template.apply<@ggml.qk.scale_min>(%weight, %base, %group) : (buffer, offset, index) -> (f32, f32)
  %output_view = buffer.view %output[%base] : buffer -> view<2xf32>
  view.store %scale, %output_view[0] : f32, view<2xf32>
  view.store %minimum, %output_view[1] : f32, view<2xf32>
  kernel.return
}
)loom";

} // namespace

loomc_status_t EmitScaleMinProbe(loomc_context_t *context,
                                 loomc_workspace_t *workspace,
                                 loomc_module_t **out_module,
                                 loomc_result_t **out_result) {
  if (out_module)
    *out_module = nullptr;
  if (out_result)
    *out_result = nullptr;
  if (!context || !workspace || !out_module || !out_result) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "probe emission requires context, workspace and outputs");
  }
  loomc_source_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
  options.structure_size = sizeof(options);
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.identifier = loomc_make_cstring_view("registry_scale_min_probe.loom");
  options.contents =
      loomc_make_byte_span(kScaleMinProbe, sizeof(kScaleMinProbe) - 1);
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t *source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  status = loomc_module_deserialize_text_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), out_module,
      out_result);
  loomc_source_release(source);
  return status;
}

loomc_status_t
LinkConsumer(loomc_context_t *context, loomc_workspace_t *workspace,
             loomc_link_index_t *libraries, const loomc_module_t *input,
             loomc_string_view_t root, loomc_module_t **out_module,
             loomc_result_t **out_result) {
  if (out_module)
    *out_module = nullptr;
  if (out_result)
    *out_result = nullptr;
  if (!context || !workspace || !libraries || !input || !root.data ||
      !root.size || !out_module || !out_result) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "consumer linking requires context, workspace, "
                             "library index, input, root and outputs");
  }
  loomc_linker_t *linker = nullptr;
  loomc_status_t status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  if (!loomc_status_is_ok(status))
    return status;
  loomc_link_module_provider_t provider = {};
  provider.module = input;
  provider.provider_name = loomc_make_cstring_view("registry.consumer");
  provider.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
  loomc_link_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.link_index = libraries;
  options.mode = LOOMC_LINK_MODE_LINK;
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  options.module_providers = &provider;
  options.module_provider_count = 1;
  status =
      loomc_link_module(linker, workspace, &options, out_module, out_result);
  loomc_linker_release(linker);
  return status;
}

} // namespace hrx::loomify_gguf
