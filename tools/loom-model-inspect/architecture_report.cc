// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loom-model-inspect/architecture_report.h"

#include <algorithm>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace hrx::model_inspect {
using model::LayerKind;
namespace {
// Lossless escaping prevents exact tensor names from injecting report lines.
std::string Quote(const std::string &s) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32 || c >= 127)
      out << "\\x" << std::hex << std::setw(2) << std::setfill('0')
          << unsigned(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
void Row(std::ostream &out, const std::string &label,
         const std::string &value) {
  out << "  " << std::left << std::setw(22) << label << value << '\n';
}
std::string HumanNumber(double value) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::defaultfloat << std::setprecision(6) << value;
  return out.str();
}
std::string WidthSummary(const std::set<uint64_t> &widths) {
  if (widths.empty())
    return "not applicable (no layers)";
  if (widths.size() == 1)
    return std::to_string(*widths.begin()) + " (uniform)";
  return "varies (min " + std::to_string(*widths.begin()) + ", max " +
         std::to_string(*widths.rbegin()) + "; see layers)";
}
struct DtypeSummary {
  std::string name;
  uint64_t count = 0;
  uint64_t bytes = 0;
};
bool AddChecked(uint64_t value, uint64_t *total) {
  if (value > std::numeric_limits<uint64_t>::max() - *total)
    return false;
  *total += value;
  return true;
}
struct BindingGroup {
  const char *title;
  std::vector<const model::TensorBinding *> bindings;
};
// One semantic order drives both displayed roles and model-wide references.
std::vector<BindingGroup>
GroupBindings(const std::vector<model::TensorBinding> &bindings,
              std::optional<LayerKind> kind = {}) {
  std::vector<BindingGroup> groups;
  std::set<size_t> handled;
  auto add = [&](const char *title, std::initializer_list<const char *> roles) {
    BindingGroup group{title, {}};
    for (const auto *role : roles)
      for (size_t i = 0; i < bindings.size(); ++i)
        if (bindings[i].role == role && handled.insert(i).second)
          group.bindings.push_back(&bindings[i]);
    if (!group.bindings.empty())
      groups.push_back(std::move(group));
  };
  if (!kind) {
    add("Embedding", {"token_embedding"});
    add("Output", {"output_norm", "output"});
    add("Position", {"rope_frequencies"});
  } else {
    if (*kind == LayerKind::kAttention)
      add("Attention", {"input_norm", "query", "key", "value", "query_norm",
                        "key_norm", "attention_output"});
    else
      add("Recurrent", {"input_norm", "recurrent_qkv", "recurrent_gate",
                        "alpha", "beta", "convolution", "dt_bias", "decay",
                        "recurrent_norm", "recurrent_output"});
    add("Feed-forward", {"post_norm", "ffn_gate", "ffn_up", "ffn_down"});
  }
  BindingGroup other{"Other", {}};
  for (size_t i = 0; i < bindings.size(); ++i)
    if (!handled.count(i))
      other.bindings.push_back(&bindings[i]);
  if (!other.bindings.empty())
    groups.push_back(std::move(other));
  return groups;
}
} // namespace

bool FormatArchitecture(const model::GgufCatalog &c,
                        const model::ModelDescriptor &m, std::string *report,
                        std::string *error, const ReportOptions &options) {
  if (!report || !error)
    return false;
  if (options.layer_index && options.tensor_name) {
    *error = "--layer and --tensor are mutually exclusive";
    return false;
  }
  if (options.layer_index &&
      std::none_of(m.layers.begin(), m.layers.end(), [&](const auto &layer) {
        return layer.index == *options.layer_index;
      })) {
    *error =
        "unknown model layer index: " + std::to_string(*options.layer_index);
    return false;
  }
  uint64_t encoded_bytes = 0;
  std::map<uint32_t, DtypeSummary> dtypes;
  std::map<std::string, const model::TensorRecord *> catalog;
  for (const auto &tensor : c.tensors) {
    if (!catalog.emplace(tensor.name, &tensor).second) {
      *error = "duplicate catalog tensor name: " + Quote(tensor.name);
      return false;
    }
    if (tensor.byte_size >
        std::numeric_limits<uint64_t>::max() - tensor.absolute_offset) {
      *error = "tensor file range end overflows uint64: " + Quote(tensor.name);
      return false;
    }
    auto &dtype = dtypes[tensor.storage_type];
    dtype.name = tensor.storage_name;
    if (!AddChecked(1, &dtype.count) ||
        !AddChecked(tensor.byte_size, &encoded_bytes) ||
        !AddChecked(tensor.byte_size, &dtype.bytes)) {
      *error = "encoded tensor byte total overflows uint64";
      return false;
    }
  }
  if (options.tensor_name && !catalog.count(*options.tensor_name)) {
    *error = "unknown tensor name: " + Quote(*options.tensor_name);
    return false;
  }
  std::vector<const model::LayerDescriptor *> layers;
  for (const auto &layer : m.layers)
    layers.push_back(&layer);
  std::stable_sort(
      layers.begin(), layers.end(),
      [](const auto *a, const auto *b) { return a->index < b->index; });
  std::map<std::string, std::vector<std::string>> references;
  uint64_t reference_count = 0;
  auto index_roles = [&](const auto &groups, const std::string &scope) {
    for (const auto &group : groups)
      for (const auto *binding : group.bindings) {
        if (!catalog.count(binding->tensor.name)) {
          *error = "role tensor is absent from catalog: " +
                   Quote(binding->tensor.name);
          return false;
        }
        if (!AddChecked(1, &reference_count)) {
          *error = "model role-reference count overflows uint64";
          return false;
        }
        references[binding->tensor.name].push_back(scope + "." + binding->role);
      }
    return true;
  };
  if (!index_roles(GroupBindings(m.globals), "global"))
    return false;
  for (const auto *layer : layers)
    if (!index_roles(GroupBindings(layer->tensors, layer->kind),
                     "layer " + std::to_string(layer->index)))
      return false;
  size_t attention_count = 0, recurrent_count = 0;
  std::set<uint64_t> ffn_widths;
  std::set<std::pair<uint64_t, uint64_t>> attention_heads;
  for (const auto &layer : m.layers) {
    ffn_widths.insert(layer.feed_forward_width);
    if (layer.kind == LayerKind::kAttention) {
      ++attention_count;
      attention_heads.emplace(layer.query_heads, layer.kv_heads);
    } else if (layer.kind == LayerKind::kGatedDeltaNet) {
      ++recurrent_count;
    }
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "Model\n";
  if (!options.source.empty())
    Row(out, "Source", Quote(options.source));
  Row(out, "Name", m.name.empty() ? "not provided" : Quote(m.name));
  Row(out, "Architecture", m.family);
  Row(out, "GGUF version", std::to_string(c.version));
  Row(out, "Layers",
      std::to_string(m.layers.size()) + " total / " +
          std::to_string(attention_count) + " attention / " +
          std::to_string(recurrent_count) + " recurrent");
  Row(out, "Embedding width", std::to_string(m.embedding_width));
  Row(out, "Feed-forward width", WidthSummary(ffn_widths));
  if (attention_heads.empty()) {
    Row(out, "Attention heads", "not applicable (no attention layers)");
  } else {
    std::string heads;
    if (attention_heads.size() == 1) {
      auto [query, kv] = *attention_heads.begin();
      heads = std::to_string(query) + " query / " + std::to_string(kv) +
              " KV (uniform)";
    } else {
      heads = "varies (" + std::to_string(attention_heads.size()) +
              " query/KV configurations; see layers)";
    }
    Row(out, "Attention heads", heads);
    Row(out, "Head widths",
        std::to_string(m.key_head_width) + " key / " +
            std::to_string(m.value_head_width) + " value");
  }
  Row(out, "Vocabulary", std::to_string(m.vocabulary_size) + " tokens");
  Row(out, "Training context", std::to_string(m.training_context) + " tokens");
  Row(out, "RMS epsilon", HumanNumber(m.rms_epsilon));
  const auto output =
      std::find_if(m.globals.begin(), m.globals.end(), [](const auto &binding) {
        return binding.role == "output";
      });
  Row(out, "Output",
      output == m.globals.end()
          ? "not bound"
          : Quote(output->tensor.name) + (m.output_tied_to_embedding
                                              ? " (tied to embedding)"
                                              : " (separate)"));
  if (attention_count && m.rope.width) {
    out << "\nPosition encoding\n";
    Row(out, "Kind", "RoPE");
    Row(out, "Width", std::to_string(m.rope.width) + " elements");
    Row(out, "Base", HumanNumber(m.rope.base));
    Row(out, "Scaling", m.rope.scaling_kind);
    Row(out, "Scaling factor", HumanNumber(m.rope.scaling_factor));
    Row(out, "Original context",
        std::to_string(m.rope.original_context) + " tokens");
    if (std::any_of(m.rope.sections.begin(), m.rope.sections.end(),
                    [](uint64_t section) { return section != 0; })) {
      std::ostringstream sections;
      sections.imbue(std::locale::classic());
      sections << '[';
      for (size_t i = 0; i < m.rope.sections.size(); ++i)
        sections << (i ? "," : "") << m.rope.sections[i];
      sections << ']';
      Row(out, "Sections", sections.str());
    }
  }
  if (recurrent_count && m.ssm) {
    const auto &ssm = *m.ssm;
    out << "\nRecurrent geometry\n";
    Row(out, "Convolution kernel",
        std::to_string(ssm.convolution_kernel) + " taps");
    Row(out, "Inner width", std::to_string(ssm.inner_width));
    Row(out, "State width", std::to_string(ssm.state_width));
    Row(out, "Heads",
        std::to_string(ssm.key_heads) + " key / " +
            std::to_string(ssm.value_heads) + " value");
    Row(out, "History",
        std::to_string(ssm.convolution_history_elements) +
            " elements / layer / sequence");
    Row(out, "Matrix state",
        std::to_string(ssm.recurrent_state_elements) +
            " elements / layer / sequence");
  }
  out << "\nStorage (file totals; encoded formats, not runtime memory)\n";
  Row(out, "Stored tensors", std::to_string(c.tensors.size()));
  Row(out, "Model-role references", std::to_string(reference_count));
  Row(out, "Metadata entries", std::to_string(c.metadata.size()));
  Row(out, "File bytes", std::to_string(c.file_size));
  Row(out, "Data offset", std::to_string(c.data_offset) + " bytes, absolute");
  Row(out, "Encoded tensor bytes",
      std::to_string(encoded_bytes) + " (sum of catalog entries)");
  for (const auto &[type, dtype] : dtypes) {
    Row(out, dtype.name + " (tag " + std::to_string(type) + ")",
        std::to_string(dtype.count) + " tensors / " +
            std::to_string(dtype.bytes) + " bytes");
  }
  out << '\n';
  std::map<std::string, std::string> shown;
  auto detail = [&](const model::TensorRecord &tensor) {
    out << "      GGML extents [";
    for (size_t i = 0; i < 4; ++i)
      out << (i ? "," : "") << tensor.normalized_extents[i];
    out << "]; storage " << tensor.storage_name << " (tag "
        << tensor.storage_type << "); encoded bytes " << tensor.byte_size
        << "; file range [" << tensor.absolute_offset << ','
        << tensor.absolute_offset + tensor.byte_size << ") bytes, absolute\n";
    out << "      References (model-wide): ";
    const auto found = references.find(tensor.name);
    if (found == references.end()) {
      out << "unassigned (no model role)";
    } else {
      for (size_t i = 0; i < found->second.size(); ++i)
        out << (i ? "; " : "") << found->second[i];
    }
    out << '\n';
  };
  auto binding = [&](const model::TensorBinding &b, const std::string &scope) {
    const auto record = catalog.find(b.tensor.name);
    if (record == catalog.end()) {
      *error = "role tensor is absent from catalog: " + Quote(b.tensor.name);
      return false;
    }
    out << "    " << b.role << " -> " << Quote(record->first);
    const auto [first, inserted] =
        shown.emplace(record->first, scope + "." + b.role);
    if (!inserted)
      out << " (alias of " << first->second
          << (scope == "global" && b.role == "output" &&
                      m.output_tied_to_embedding
                  ? "; tied to embedding"
                  : "")
          << ')';
    out << '\n';
    if (inserted)
      detail(*record->second);
    return true;
  };
  auto render_groups = [&](const auto &groups, const std::string &scope) {
    for (const auto &group : groups) {
      out << "  " << group.title << '\n';
      for (const auto *entry : group.bindings)
        if (!binding(*entry, scope))
          return false;
    }
    return true;
  };
  if (options.tensor_name) {
    const auto &tensor = *catalog.at(*options.tensor_name);
    out << "Tensor details (GGML order; four padded extents)\n"
        << "    tensor " << Quote(tensor.name) << '\n';
    shown.emplace(tensor.name, "tensor");
    detail(tensor);
  } else {
    out << "Global tensors (GGML order; four padded extents)\n";
    if (!render_groups(GroupBindings(m.globals), "global"))
      return false;
    out << "\nLayer details: " << (options.layer_index ? 1 : m.layers.size())
        << " of " << m.layers.size()
        << " model layers (ordinals are logical, not cache offsets)\n";
    for (const auto *layer : layers) {
      const auto &d = *layer;
      if (options.layer_index && d.index != *options.layer_index)
        continue;
      out << "\nLayer " << d.index << ' '
          << (d.kind == LayerKind::kAttention ? "attention" : "gated_delta_net")
          << '\n';
      Row(out, "Model index", std::to_string(d.index));
      const auto ordinal = d.kind == LayerKind::kAttention
                               ? d.attention_ordinal
                               : d.recurrent_ordinal;
      Row(out,
          d.kind == LayerKind::kAttention ? "Attention ordinal"
                                          : "Recurrent ordinal",
          ordinal ? std::to_string(*ordinal) + " (logical)" : "not provided");
      Row(out, "Feed-forward width", std::to_string(d.feed_forward_width));
      if (d.kind == LayerKind::kAttention) {
        Row(out, "Attention heads",
            std::to_string(d.query_heads) + " query / " +
                std::to_string(d.kv_heads) + " KV");
        Row(out, "Head widths",
            std::to_string(m.key_head_width) + " key / " +
                std::to_string(m.value_head_width) + " value");
      } else if (m.ssm) {
        Row(out, "Recurrent heads",
            std::to_string(m.ssm->key_heads) + " key / " +
                std::to_string(m.ssm->value_heads) + " value");
        Row(out, "Recurrent geometry", "see model summary (SSM)");
      } else {
        Row(out, "Recurrent geometry", "not provided");
      }
      const auto scope = "layer " + std::to_string(d.index);
      if (!render_groups(GroupBindings(d.tensors, d.kind), scope))
        return false;
    }
    if (!options.layer_index) {
      bool heading = false;
      for (const auto &tensor : c.tensors) {
        if (shown.count(tensor.name))
          continue;
        if (!heading) {
          out << "\nUnassigned tensors (no model role)\n";
          heading = true;
        }
        out << "    tensor " << Quote(tensor.name) << '\n';
        shown.emplace(tensor.name, "unassigned");
        detail(tensor);
      }
    }
  }
  uint64_t shown_bytes = 0, shown_references = 0;
  for (const auto &[name, first_role] : shown) {
    if (!AddChecked(catalog.at(name)->byte_size, &shown_bytes) ||
        !AddChecked(references[name].size(), &shown_references)) {
      *error = "shown inventory total overflows uint64";
      return false;
    }
  }
  out << "\nView\n";
  Row(out, "Shown unique tensors", std::to_string(shown.size()));
  Row(out, "Shown encoded bytes", std::to_string(shown_bytes));
  Row(out, "Shown model-role refs", std::to_string(shown_references));
  out << "Metadata bytes read: " << c.bytes_read
      << " (includes header validation; no tensor payload)\n";
  *report = out.str();
  return true;
}
} // namespace hrx::model_inspect
