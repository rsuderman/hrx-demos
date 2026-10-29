// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LOOMIFY_PIPELINE_BUILDER_H_
#define HRX_SOLUTIONS_LOOMIFY_PIPELINE_BUILDER_H_

#include <functional>
#include <string_view>
#include <vector>

#include "loom/ir/memory_space.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

namespace hrx::loomify_gguf {

// Native construction helpers for use within ModuleBuilder::BuildNative. All
// native types, values and references are borrowed from that callback's module;
// none may be retained across module finalization, cloning or destruction.
class PipelineBody;
using PipelineBodyCallback = std::function<iree_status_t(PipelineBody &)>;

class PipelineRef {
public:
  PipelineRef() = default;

private:
  friend class PipelineBody;
  friend class PipelineBuilder;
  loom_module_t *module_ = nullptr;
  loom_symbol_ref_t symbol_ = {};
};

struct PipelineInterface {
  std::string_view name;
  std::vector<loom_type_t> specializations;
  std::vector<loom_type_t> run;
  // Optional type factory for run types depending on freshly created leading
  // specialization arguments. Mutually exclusive with nonempty run.
  std::function<iree_status_t(loom_module_t *,
                              const std::vector<loom_value_id_t> &,
                              std::vector<loom_type_t> *)>
      dependent_run;
  bool kernel_scope = false;
  bool is_public = false;
  // Empty means inherited/no target. A named target must already be declared.
  std::string_view target;
};

struct PipelineWorkers {
  std::vector<int64_t> origins;
  std::vector<int64_t> counts;
  std::vector<int64_t> strides;
};

struct ChannelAccess {
  loom_value_id_t ownership = LOOM_VALUE_ID_INVALID;
  loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
};

class PipelineBody {
public:
  PipelineBody(const PipelineBody &) = delete;
  PipelineBody &operator=(const PipelineBody &) = delete;
  loom_value_id_t Specialization(size_t index) const;
  loom_value_id_t Binding(size_t index) const;

  // Regions declare independently progressing workers. The callback is called
  // synchronously while constructing IR, not while executing the pipeline.
  iree_status_t Strand(const PipelineWorkers &workers,
                       const PipelineBodyCallback &body);
  iree_status_t Compose(PipelineRef child,
                        const std::vector<loom_value_id_t> &specializations,
                        const std::vector<loom_value_id_t> &bindings);

  iree_status_t IndexConstant(int64_t value, loom_value_id_t *out);
  iree_status_t OffsetConstant(int64_t value, loom_value_id_t *out);
  iree_status_t MemoryPool(loom_value_fact_memory_space_t space,
                           const std::vector<int64_t> &coordinates,
                           loom_value_id_t *out);
  iree_status_t Allocate(loom_value_id_t pool,
                         loom_value_fact_memory_space_t space,
                         int64_t alignment, loom_value_id_t bytes,
                         loom_value_id_t *out);
  iree_status_t View(loom_value_id_t buffer, loom_value_id_t byte_offset,
                     loom_type_t type, loom_value_id_t *out);
  iree_status_t BindFifo(loom_value_id_t slots, loom_value_id_t capacity,
                         loom_type_t payload, loom_value_id_t *out);
  iree_status_t Reserve(loom_value_id_t channel, ChannelAccess *out);
  iree_status_t Publish(loom_value_id_t write);
  iree_status_t Acquire(loom_value_id_t channel, ChannelAccess *out);
  iree_status_t Release(loom_value_id_t read);

  // Existing native numerical builders may write the current construction
  // region. Return their errors from the callback to fail the module owner.
  loom_builder_t *Native() { return &builder_; }
  loom_module_t *Module() { return module_; }

private:
  friend class PipelineBuilder;
  PipelineBody(loom_module_t *module, loom_builder_t builder,
               const std::vector<loom_value_id_t> &args, size_t specializations,
               bool strand);
  iree_status_t ChannelTypes(loom_value_id_t channel, loom_type_t *ownership,
                             loom_type_t *view, bool write);
  loom_module_t *module_;
  loom_builder_t builder_;
  std::vector<loom_value_id_t> args_;
  size_t specializations_;
  bool strand_;
};

class PipelineBuilder {
public:
  // The callback must return every construction error. Definition and strand
  // terminators are appended automatically only when their callbacks succeed.
  static iree_status_t Define(loom_module_t *module, loom_builder_t *top,
                              const PipelineInterface &interface,
                              const PipelineBodyCallback &body,
                              PipelineRef *out);
};

} // namespace hrx::loomify_gguf

#endif // HRX_SOLUTIONS_LOOMIFY_PIPELINE_BUILDER_H_
