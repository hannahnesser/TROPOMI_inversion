#!/bin/bash

PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_????"
DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

sbatch --array=1 build_k_monthly.sh ${PRIOR_DIR} ${PERT_DIRS@Q} ${DATA_DIR} ${CODE_DIR}
