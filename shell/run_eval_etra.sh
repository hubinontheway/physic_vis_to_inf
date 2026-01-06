#!/bin/bash

# Evaluation script for ETRA flat config
# Note: eval_etra.py can take --config or --run-dir. 
# Using --config will automatically resolve the latest run directory if it exists.
python3 eval_etra.py --config configs/etra/etra_flat.yml
