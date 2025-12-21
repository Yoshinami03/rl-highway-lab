#!/bin/bash
cd "$(dirname "$0")"
source HighwayEnv_Merge/bin/activate
python3 src/visualize_random.py "$@"

