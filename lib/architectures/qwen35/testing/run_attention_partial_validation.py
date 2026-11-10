# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Finite actual attention3 partial-active witness using the retained harness.

The existing caller-owned attended buffer is poisoned and inspected; no private
scratch allocation is exposed. Cross-T bitwise comparisons require the reviewed
same H256 WMMA route, not an assumption about different precision policies.
"""
from run_gdn_partial_validation import main

if __name__ == '__main__':
    main('attention3')
