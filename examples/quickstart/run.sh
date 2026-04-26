#!/usr/bin/env bash
# Quickstart — compile the notes into adapters and inspect them.
set -euo pipefail
cd "$(dirname "$0")"

llm-patch compile ./notes --output ./out
llm-patch adapter status ./out
