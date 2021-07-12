#!/bin/bash

# User settings
PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/"
TROPOMI_DIR="/n/seasasfs02/CH4_inversion/InputData/Obs/TROPOMI/"
CORRECT_CFL_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion/gc_outputs/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"
OUTPUT_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data"

# Check for unphysical stratospheric values
jid1=$(sbatch preprocess_GC_output.sh ${PRIOR_DIR} ${CORRECT_CFL_DIR} ${CODE_DIR})

# Apply the TROPOMI operator
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-12 apply_TROPOMI_operator.sh ${TROPOMI_DIR} ${PRIOR_DIR})
jid2=$(sbatch --array=1 apply_TROPOMI_operator.sh ${TROPOMI_DIR} ${PRIOR_DIR})

# Analyze the output
sbatch --dependency=afterok:${jid2##* } run_generate_obs.sh ${PRIOR_DIR} ${OUTPUT_DIR} ${CODE_DIR}
# sbatch run_generate_obs.sh ${PRIOR_DIR} ${OUTPUT_DIR} ${CODE_DIR}
