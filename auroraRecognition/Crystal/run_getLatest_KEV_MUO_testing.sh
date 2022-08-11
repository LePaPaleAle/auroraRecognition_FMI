#!/bin/bash -x
# Activate the Conda environment
# The paths in this script must be checked when moved to another location!
source ~/miniforge3/bin/activate condaEnv
# Change to correct directory
cd ~/testing
# Run the get Latest scripts
python3 getLatestKEV_testing.py
python3 getLatestMUO_testing.py
# Deactivate the conda env
conda deactivate
