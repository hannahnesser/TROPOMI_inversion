#!/bin/bash

#SBATCH -J TROPOMI_operator
#SBATCH -o %x_%j_%a.out
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 8000
#SBATCH -t 0-00:40
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
CODE_DIR="${1}"
TROPOMI_DATA_DIR="${2}"
PRIOR_DIR="${3}"
RUN_DIR="${4}"
JACOBIAN="${5}"

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
python_dir=$(dirname `pwd`)
cd $INPUT_DIR
mkdir -p $OUTPUT_DIR

echo "Initiating script"
python -u ${python_dir}/python/TROPOMI_operator.py $CODE_DIR $TROPOMI_DATA_DIR $PRIOR_DIR $RUN_DIR $MONTH $JACOBIAN
