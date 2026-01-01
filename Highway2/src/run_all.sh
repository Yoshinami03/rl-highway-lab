#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python Run.py
python Learn.py
python AdditionalLearn.py
python Test.py
python Draw.py