#!/bin/bash

PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_NNNN"
NPERT_DIRS=1952
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results"
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"
NUM_EVECS="2613"
CALCULATE_EVECS="True"
FORMAT_EVECS="False"
OPTIMIZE_RF="False"
CHUNK_SIZE=150000
RF="1"
SA="1"

# Build the Jacobian
jid1=$(sbatch --array=1-20 build_k_chunks.sh ${CHUNK_SIZE} "2" ${PRIOR_DIR} ${PERT_DIRS} ${NPERT_DIRS} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Calculate the prior preconditioned Hessian
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-20 generate_pph.sh ${CHUNK_SIZE} "2" ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# jid2=$(sbatch --array=1-20 generate_pph.sh ${CHUNK_SIZE} "2" ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Calculate the eigenvectors
jid3=$(sbatch --dependency=afterok:${jid2##* } generate_evecs.sh "2" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR} ${CALCULATE_EVECS} ${FORMAT_EVECS})
# jid3=$(sbatch generate_evecs.sh "2" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR} ${CALCULATE_EVECS} ${FORMAT_EVECS})

# Solve the inversion
jid4=$(sbatch --dependency=afterok:${jid3##* } run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR} ${OPTIMIZE_RF} ${RF} ${SA})
# jid4=$(sbatch run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR} ${OPTIMIZE_RF} ${RF} ${SA})
