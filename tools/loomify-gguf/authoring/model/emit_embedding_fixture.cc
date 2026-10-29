// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model/embedding_fixture.h"
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "usage: emit_embedding_fixture FIXTURE.gguf CHECKS.loom\n");
    return 1;
  }
  const auto bytes = hrx::loomify_gguf::testing::EmbeddingGguf();
  std::vector<float> table;
  std::string error;
  if (!hrx::loomify_gguf::testing::ReadNamedF32(bytes, "token_embd.weight",
                                                &table, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  FILE *file = std::fopen(argv[1], "wb");
  if (!file)
    return 1;
  bool written =
      std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
  if (std::fclose(file) != 0 || !written)
    return 1;
  std::ostringstream checks;
  checks
      << "// Synthetic GGUF index -> packed immutable parameter root.\n"
         "command.program.decl @model.embedding(%token: index) "
         "launch(%parameters: buffer, %output: buffer)\n"
         "func.def @initialize_embedding_parameters(%token: index, "
         "%parameters: buffer) {\n"
         "  %base = index.constant 0 : offset\n"
         "  %table = buffer.view %parameters[%base] : buffer -> view<64xf32>\n";
  for (size_t i = 0; i < table.size(); ++i) {
    checks << "  %v" << i << " = scalar.constant " << std::fixed
           << std::setprecision(1) << table[i] << " : f32\n"
           << "  view.store %v" << i << ", %table[" << i
           << "] : f32, view<64xf32>\n";
  }
  checks << R"loom(  %token_i32 = index.cast %token : index to i32
  %token_f32 = scalar.sitofp %token_i32 : i32 to f32
  view.store %token_f32, %table[63] : f32, view<64xf32>
  func.return
}
// Independent oracle uses a known mathematical fixture, never the parameter
// tensor, generated kernel, table lookup, GGUF parser or emitter's loaded values.
func.def @embedding_oracle(%parameters: buffer, %output: buffer) {
  %base = index.constant 0 : offset
  %configuration = buffer.view %parameters[%base] : buffer -> view<64xf32>
  %token_f32 = view.load %configuration[63] : view<64xf32> -> f32
  %token_i32 = scalar.fptosi %token_f32 : f32 to i32
  %token = index.cast %token_i32 : i32 to index
  %target = buffer.view %output[%base] : buffer -> view<4xf32>
  %begin = index.constant 0 : index
  %end = index.constant 4 : index
  %step = index.constant 1 : index
  %vocab = index.constant 3 : index
  %valid = index.cmp ult, %token, %vocab : index
  scf.for %column = [%begin to %end step %step] {
    %expected = scf.if %valid -> (f32) {
      %hundred = index.constant 100 : index
      %ten = index.constant 10 : index
      %row = index.mul %token, %hundred : index
      %first = index.add %row, %column : index
      %integer = index.add %first, %ten : index
      %i32 = index.cast %integer : index to i32
      %value = scalar.sitofp %i32 : i32 to f32
      scf.yield %value : f32
    } else {
      %zero = scalar.constant 0.0 : f32
      scf.yield %zero : f32
    }
    view.store %expected, %target[%column] : f32, view<4xf32>
    scf.yield
  }
  func.return
}
check.scenario public @gguf_embedding_rows configure[4](%token: index, %configuration_entropy: check.entropy) {
  check.trial[1](%trial: index, %entropy: check.entropy) {
    %parameter_owner = check.generate.fill value(-777.0) : tensor<144xf32>
    %parameters = check.tensor.view %parameter_owner offset(256) : tensor<144xf32> -> tensor<64xf32>
    check.generate<@initialize_embedding_parameters>(%token, %parameters) : (index, tensor<64xf32>) -> ()
    %output_owner = check.generate.fill value(-999.0) : tensor<8xf32>
    %output = check.tensor.view %output_owner offset(8) : tensor<8xf32> -> tensor<4xf32>
    check.compare<@model.embedding, @embedding_oracle>[%token](%parameters, %output) : [index](tensor<64xf32>, tensor<4xf32>) -> () {
      check.expect.bitwise actual(%parameter_owner) expected(%parameter_owner) : tensor<144xf32>
      check.expect.bitwise actual(%output_owner) expected(%output_owner) : tensor<8xf32>
    }
  }
  check.return
}
)loom";
  const std::string text = checks.str();
  file = std::fopen(argv[2], "wb");
  if (!file)
    return 1;
  written = std::fwrite(text.data(), 1, text.size(), file) == text.size();
  return std::fclose(file) == 0 && written ? 0 : 1;
}
