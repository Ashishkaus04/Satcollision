#!/bin/bash
# Time secure_pc under several 3-party protocols (run inside MP-SPDZ's directory).
# Usage: bash run_protocols.sh <n> <protocol> [<protocol> ...]
n=$1; shift
for p in "$@"; do
  echo "== $p"
  timeout 1800 Scripts/$p.sh secure_pc-$n-show$SUFFIX 2>&1 | grep -E "bits|Time =|Global|rror" | sed 's/bits \[\(.\{40\}\).*/bits [\1 .../'
done
