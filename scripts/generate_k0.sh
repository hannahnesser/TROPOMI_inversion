#!/bin/bash

# Directories
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion/"
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python/"

# First, generate the nstate x nstate x months first guess of the Jacobian.
# Note that there are some directory relationships that are hard-coded here--
# so be careful about changes to directory structures.
# Command structure:
# sbatch generate_k0_nstate data_dir output_dir code_dir
jid1=$(sbatch generate_k0_nstate.sh ${LONG_TERM_DATA_DIR} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Second, generate the first guess of the m x n Jacobian on a monthly basis
# Command structure:
# sbatch generate_k0_monthly.sh data_dir output_dir code_dir
jid2=$(sbatch --dependency=afterok:${jid1##* } generate_k0_monthly.sh ${LONG_TERM_DATA_DIR} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# jid2=$(sbatch generate_k0_monthly.sh ${LONG_TERM_DATA_DIR} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Third, generate the monthly prior pre-conditioned Hessians.
# Command structure:
# sbatch generate_k0_nstate data_dir code_dir
jid3=$(sbatch --dependency=afterok:${jid2##* } --array=1-12 generate_pph0.sh ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
# sbatch --array=1-12 generate_pph0.sh ${SHORT_TERM_DATA_DIR} ${CODE_DIR}

# Fourth and finally, calculate the eigenvectors
jid4=$(sbatch --dependency=afterok:${jid3##* } generate_evecs0.sh ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${CODE_DIR})
# sbatch generate_evecs0.sh ${SHORT_TERM_DATA_DIR} ${CODE_DIR}
