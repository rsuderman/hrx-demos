// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Computation source consumed by Loom's existing C++ importer. The host command
// program is authored separately; this helper remains numerical computation.
#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

[[loom::device, loom::symbol("motif.affine")]]
static int affine_motif(int value) {
  return value * 3 + 2;
}

[[loom::kernel, loom::symbol("authored.affine"), loom::workgroup_count(7, 1, 1),
  loom::workgroup_size(1, 1, 1)]]
void authored_affine(loom::type::buffer<int> input,
                     loom::type::buffer<int> output) {
  unsigned position = loom::kernel::workgroup::id.x;
  if (position < 7u) {
    unsigned index = position + 2u;
    auto source =
        loom::buffer::view<11>(input, {}, loom::encoding::layout::dense<1>());
    auto destination =
        loom::buffer::view<11>(output, {}, loom::encoding::layout::dense<1>());
    loom::view::store(affine_motif(loom::view::load(source, index)),
                      destination, index);
  }
}
