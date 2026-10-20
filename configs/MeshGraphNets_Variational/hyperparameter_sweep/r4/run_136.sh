#!/usr/bin/env bash
# B300 round 4, node 136: see run_node.sh (order of work, overrides) and README.md (the 8 jobs).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_node.sh" 136
