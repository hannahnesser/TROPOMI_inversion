#!/bin/bash

PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_????"
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion"
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

# Build the Jacobian
jid1=$(sbatch --array=1-12 build_k_monthly.sh ${PRIOR_DIR} ${PERT_DIRS} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Calculate the prior preconditioned Hessian
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-12 generate_pph.sh "1" ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# sbatch --array=2-12 generate_pph.sh "1" ${SHORT_TERM_DATA_DIR} ${CODE_DIR}

# Calculate the eigenvectors
jid3=$(sbatch --dependency=afterok:${jid2##* } generate_evecs.sh "1" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR})
# jid3=$(sbatch --dependency=afterok:48100973 generate_evecs.sh "1" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR})
# sbatch generate_evecs.sh "1" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR}
