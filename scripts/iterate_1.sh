#!/bin/bash

PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs0/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs0/TROPOMI_inversion_NNNN"
NPERT_DIRS=434
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results"
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"
NUM_EVECS="2613"
CALCULATE_EVECS="True"
FORMAT_EVECS="True"
SOLVE_INVERSION="False"
CHUNK_SIZE=125000

# Build the Jacobian
jid1=$(sbatch --array=1-24 build_k_chunks.sh ${CHUNK_SIZE} "1" ${PRIOR_DIR} ${PERT_DIRS} ${NPERT_DIRS} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# j1d1=$(sbatch --array=6 build_k_monthly.sh "1" ${PRIOR_DIR} ${PERT_DIRS} ${NPERT_DIRS} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Calculate the prior preconditioned Hessian
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-24 generate_pph.sh ${CHUNK_SIZE} "1" ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# sbatch --array=8 generate_pph.sh "1" ${SHORT_TERM_DATA_DIR} ${CODE_DIR}

# Calculate the eigenvectors
jid3=$(sbatch --dependency=afterok:${jid2##* } generate_evecs.sh "1" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR} ${CALCULATE_EVECS} ${FORMAT_EVECS} ${SOLVE_INVERSION})
# jid3=$(sbatch --dependency=afterok:51606797 generate_evecs.sh "1" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR})
# sbatch generate_evecs.sh "1" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR}
