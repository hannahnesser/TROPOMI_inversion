#!/bin/bash

#SBATCH -J TROPOMI_operator
#SBATCH -o %x_%j_%a.out
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 8000
#SBATCH -t 0-02:00
##SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Time
MONTH="${SLURM_ARRAY_TASK_ID}"
CODE_DIR=${1}

## -------------------------------------------------------------------------##
## Load the environment
## -------------------------------------------------------------------------##
module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion
echo "Activated ${CONDA_PREFIX}"

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"
python -u ${CODE_DIR}/TROPOMI_operator.py ${@} $MONTH
