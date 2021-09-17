#!/bin/bash

#SBATCH -J TROPOMI_operator
#SBATCH -o %x_%j_%a.out
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 8000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
CODE_DIR="${1}"
TROPOMI_DATA_DIR="${2}"
PRIOR_DIR="${3}"
RUN_DIR="${4}"
OUTPUT_DIR="${5}"
JACOBIAN="${6}"

# Time
MONTH="${SLURM_ARRAY_TASK_ID}"

## -------------------------------------------------------------------------##
## Load the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"
python -u ${CODE_DIR}/TROPOMI_operator.py $CODE_DIR $TROPOMI_DATA_DIR $PRIOR_DIR $RUN_DIR $OUTPUT_DIR $MONTH $JACOBIAN
