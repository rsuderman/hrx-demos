// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "tools/loom-model-inspect/architecture_report.h"
#include "tools/model/model_descriptor.h"

IREE_FLAG(string, gguf, nullptr,
          "Load and report the GGUF architecture without loading tensor data.");
IREE_FLAG(string, layer, nullptr,
          "Show one model layer index (nonnegative decimal), with globals and "
          "summary.");
IREE_FLAG(string, tensor, nullptr,
          "Show one exact raw tensor name with all model role references.");

int main(int argc, char **argv) {
  // HRX flags use --name=value. Also accept separate values for these options.
  std::vector<std::string> arguments;
  bool has_layer = false;
  bool has_tensor = false;
  std::string layer_value;
  std::string tensor_value;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--layer") == 0 ||
        std::strncmp(argv[i], "--layer=", 8) == 0) {
      if (has_layer) {
        std::fprintf(stderr, "loom-model-inspect: repeated --layer option\n");
        return 1;
      }
      has_layer = true;
      if (argv[i][7] == '=')
        layer_value = argv[i] + 8;
    }
    if (std::strcmp(argv[i], "--tensor") == 0 ||
        std::strncmp(argv[i], "--tensor=", 9) == 0) {
      if (has_tensor) {
        std::fprintf(stderr, "loom-model-inspect: repeated --tensor option\n");
        return 1;
      }
      has_tensor = true;
      if (argv[i][8] == '=')
        tensor_value = argv[i] + 9;
    }
    if (std::strcmp(argv[i], "--gguf") == 0 ||
        std::strcmp(argv[i], "--layer") == 0 ||
        std::strcmp(argv[i], "--tensor") == 0) {
      const std::string option = argv[i];
      if (i + 1 == argc || std::strncmp(argv[i + 1], "--", 2) == 0) {
        std::fprintf(stderr, "loom-model-inspect: %s requires a value\n",
                     option.c_str());
        return 1;
      }
      const std::string value = argv[++i];
      if (option == "--layer")
        layer_value = value;
      if (option == "--tensor")
        tensor_value = value;
      arguments.push_back(option + "=" + value);
    } else {
      arguments.emplace_back(argv[i]);
    }
  }
  if (has_layer && has_tensor) {
    std::fprintf(
        stderr,
        "loom-model-inspect: --layer and --tensor are mutually exclusive\n");
    return 1;
  }
  std::vector<char *> pointers;
  for (auto &argument : arguments)
    pointers.push_back(argument.data());
  pointers.push_back(nullptr);
  argc = static_cast<int>(arguments.size());
  argv = pointers.data();
  iree_flags_set_usage(
      "loom-model-inspect",
      "Load and report GGUF architecture and tensor information.\n"
      "\n"
      "Usage: loom-model-inspect --gguf=PATH [--layer=N | "
      "--tensor=EXACT_NAME]\n"
      "\n"
      "Architecture loading validates structure without reading tensor "
      "data.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1) {
    std::fprintf(stderr,
                 "loom-model-inspect: unexpected argument '%s'; see --help\n",
                 argv[1]);
    return 1;
  }
  if (!FLAG_gguf || FLAG_gguf[0] == '\0') {
    std::fprintf(stderr, "loom-model-inspect: --gguf requires a file path\n");
    return 1;
  }
  hrx::model_inspect::ReportOptions options;
  options.source = FLAG_gguf;
  if (has_tensor)
    options.tensor_name = tensor_value;
  if (has_layer) {
    // Validate the raw value: the generic flag parser trims whitespace.
    const std::string &value = layer_value;
    uint32_t index;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), index);
    if (value.empty() || parsed.ec != std::errc() ||
        parsed.ptr != value.data() + value.size()) {
      std::fprintf(stderr, "loom-model-inspect: --layer requires a nonnegative "
                           "decimal uint32 model index\n");
      return 1;
    }
    options.layer_index = index;
  }

  std::unique_ptr<FILE, decltype(&std::fclose)> file(
      std::fopen(FLAG_gguf, "rb"), std::fclose);
  if (!file) {
    std::fprintf(stderr, "loom-model-inspect: cannot open '%s': %s\n",
                 FLAG_gguf, std::strerror(errno));
    return 1;
  }
  // Prevent stdio from prefetching tensor payload beyond metadata reads.
  if (std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0) {
    std::fprintf(stderr,
                 "loom-model-inspect: cannot disable input buffering\n");
    return 1;
  }
  hrx::model::GgufCatalog catalog;
  hrx::model::ModelDescriptor model;
  std::string error;
  if (!hrx::model::ReadGgufCatalog(file.get(), &catalog, &error) ||
      !hrx::model::LoadModelDescriptor(catalog, &model, &error)) {
    std::fprintf(stderr, "loom-model-inspect: '%s': %s\n", FLAG_gguf,
                 error.c_str());
    return 1;
  }
  std::string report;
  if (!hrx::model_inspect::FormatArchitecture(catalog, model, &report, &error,
                                              options)) {
    std::fprintf(stderr, "loom-model-inspect: %s\n", error.c_str());
    return 1;
  }
  if (std::fwrite(report.data(), 1, report.size(), stdout) != report.size() ||
      std::fflush(stdout) != 0) {
    std::perror("loom-model-inspect: writing report");
    return 1;
  }
  return 0;
}
