#!/bin/bash

# User settings
PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/"
TROPOMI_DIR="/n/seasasfs02/CH4_inversion/InputData/Obs/TROPOMI/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

# Latitude and longitude range
LONS="-130 -60 0.3125"
LATS="9.75 60 0.25"
BUFFER="3 3 3 3"

# Time range
YEAR="2019"
# MONTH= months are set in the array variable

# Apply the TROPOMI operator
jid=$(sbatch --array=12-12 apply_TROPOMI_operator.sh ${TROPOMI_DIR} ${PRIOR_DIR})

# Analyze the output
sbatch --job-name=consolidate_obs_data \
       --output=%x_%j_%a.out \
       -c 4 -N 1 -p huce_intel --mem 12000 -t 0-00:40 \
       --dependency=afterok:${jid##* } \
       --mail-type=END \
       python generate_obs.py ${PRIOR_DIR} ${CODE_DIR}
