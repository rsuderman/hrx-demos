// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "tools/loomify-gguf/gguf_info.h"
#include "tools/loomify-gguf/source_builder.h"

IREE_FLAG(string, gguf, nullptr,
          "Inspect a GGUF model catalog without loading tensor data.");

int main(int argc, char **argv) {
  // HRX flags use --name=value. Also accept the documented --gguf PATH form.
  std::vector<std::string> arguments;
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--gguf") == 0) {
      if (i + 1 == argc || std::strncmp(argv[i + 1], "--", 2) == 0) {
        std::fprintf(stderr, "loomify-gguf: --gguf requires a file path\n");
        return 1;
      }
      arguments.push_back(std::string("--gguf=") + argv[++i]);
    } else {
      arguments.emplace_back(argv[i]);
    }
  }
  std::vector<char *> pointers;
  for (auto &argument : arguments)
    pointers.push_back(argument.data());
  pointers.push_back(nullptr);
  argc = static_cast<int>(arguments.size());
  argv = pointers.data();
  iree_flags_set_usage(
      "loomify-gguf",
      "Inspect GGUF model information or print an empty loomcxx model entry.\n"
      "\n"
      "Usage: loomify-gguf > model.cxx\n"
      "       loomify-gguf --gguf=PATH\n"
      "\n"
      "GGUF inspection validates basic structure; it does not execute or "
      "qualify\n"
      "a model. With no arguments, print the original empty C++ source.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1) {
    std::fprintf(stderr, "loomify-gguf: unexpected argument '%s'; see --help\n",
                 argv[1]);
    return 1;
  }

  std::string source;
  if (FLAG_gguf) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(
        std::fopen(FLAG_gguf, "rb"), std::fclose);
    if (!file) {
      std::fprintf(stderr, "loomify-gguf: cannot open '%s': %s\n", FLAG_gguf,
                   std::strerror(errno));
      return 1;
    }
    // Do not let stdio prefetch tensor payload past the catalog's last read.
    if (std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0) {
      std::fprintf(stderr, "loomify-gguf: cannot disable input buffering\n");
      return 1;
    }
    hrx::loomify_gguf::GgufInfo info;
    std::string error;
    if (!hrx::loomify_gguf::ReadGgufInfo(file.get(), &info, &error)) {
      std::fprintf(stderr, "loomify-gguf: '%s': %s\n", FLAG_gguf,
                   error.c_str());
      return 1;
    }
    source = hrx::loomify_gguf::FormatGgufInfo(info);
  } else {
    source = hrx::loomify_gguf::EmitEmptyProgram();
  }
  if (std::fwrite(source.data(), 1, source.size(), stdout) != source.size() ||
      std::fflush(stdout) != 0) {
    std::perror("loomify-gguf: writing source");
    return 1;
  }
  return 0;
}
