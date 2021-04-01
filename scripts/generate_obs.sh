#!/bin/bash

# User settings
PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/"
TROPOMI_DIR="/n/seasasfs02/CH4_inversion/InputData/Obs/TROPOMI/"
CORRECT_CFL_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion/gc_outputs/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

# Latitude and longitude range
LONS="-130 -60 0.3125"
LATS="9.75 60 0.25"
BUFFER="3 3 3 3"

# Time range
YEAR="2019"
# MONTH= months are set in the array variable

# Check for unphysical stratospheric values
jid1=$(sbatch correct_CFL_errors.sh ${PRIOR_DIR} ${CORRECT_CFL_DIR} ${CODE_DIR})

# Apply the TROPOMI operator
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-12 apply_TROPOMI_operator.sh ${TROPOMI_DIR} ${PRIOR_DIR})

# Analyze the output
sbatch --dependency=afterok:${jid2##* } run_generate_obs.sh ${PRIOR_DIR} ${CODE_DIR}
# sbatch run_generate_obs.sh ${PRIOR_DIR} ${CODE_DIR}
