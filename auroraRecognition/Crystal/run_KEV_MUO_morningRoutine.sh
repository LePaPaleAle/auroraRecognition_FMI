#!/bin/bash -x
#Activate the Conda environment
# The paths in this script must be checked when moved to another location!
source ~/miniforge3/bin/activate condaEnv
# Change to correct directory
cd ~/testing
# Run the morning routines
python3 morningRoutineKEV.py
python3 morningRoutineMUO.py
# Deactivate the conda env
conda deactivate
