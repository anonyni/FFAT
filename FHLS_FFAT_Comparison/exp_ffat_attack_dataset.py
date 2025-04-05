#!/usr/bin/python3
"""
Experiment script to run fuzzing using real sensor datasets.
This script loads an HJSON configuration file (which should set use_real_dataset to true)
and then calls the fuzz() function in fuzz.py.
"""

import sys
from hwfp.fuzz import fuzz

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: exp_ffat_attack_dataset.py <FFAT_experiment_config.hjson>")
        sys.exit(1)
    # Pass all arguments (config file, etc.) to fuzz()
    fuzz(sys.argv[1:])
