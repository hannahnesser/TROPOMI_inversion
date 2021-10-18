#!/bin/bash

# Directories
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion/"
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python/"


jid1=$(sbatch --dependency=afterok:${jid2##* } --array=1-12 generate_pph.sh "pph1" ${SHORT_TERM_DATA_DIR} ${CODE_DIR})
