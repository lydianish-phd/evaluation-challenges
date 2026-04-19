#!/usr/bin/env bash

# Get repo root (parent of slurm/)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$REPO_DIR/data"
OUT_DIR="$REPO_DIR/data_extracted"

mkdir -p "$OUT_DIR"

for f in "$DATA_DIR"/*.tar.gz; do
  corpus=$(basename "$f" .tar.gz)
  mkdir -p "$OUT_DIR"
  tar -xzf "$f" -C "$OUT_DIR" --overwrite
done