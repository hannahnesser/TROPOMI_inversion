#!/bin/bash

# Directories
DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python/"

rm -rf ${DATA_DIR}dask-worker-space

# First, generate the monthly prior pre-conditioned Hessians.
# Command structure:
# sbatch generate_k0_nstate data_dir code_dir
jid1=$(sbatch --array=1-3 generate_pph0.sh ${DATA_DIR} ${CODE_DIR})
# sbatch --array=1 generate_pph0.sh ${DATA_DIR} ${CODE_DIR}

# # Second, generate the first guess of the m x n Jacobian on a monthly basis
# # Command structure:
# # sbatch generate_k0_monthly.sh data_dir output_dir code_dir
# jid2=$(sbatch --dependency=afterok:${jid1##* } generate_k0_monthly.sh ${LONG_TERM_DATA_DIR} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
