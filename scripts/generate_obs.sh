#!/bin/bash 

prior_dir="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/"
jid=$(sbatch --array=12-12 apply_TROPOMI_operator.sh ${prior_dir})

${jid##* }
