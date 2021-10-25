#!/bin/bash

#SBATCH -J preprocess_GC
#SBATCH -o %x_%j.out
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 4000
#SBATCH -t 0-02:00
##SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
CODE_DIR="${1}"
DATA_DIR="${2}"
CORRECT_CFL_DIR="${3}" # Half step outputs

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
python -u ${CODE_DIR}/preprocess_GC_output.py $CODE_DIR $CORRECT_CFL_DIR $DATA_DIR
