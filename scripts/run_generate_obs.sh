#!/bin/bash

#SBATCH -J generate_obs
#SBATCH -o %x_%j.out
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 12000
#SBATCH -t 0-00:40
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
PRIOR_DIR="${1}"
OUTPUT_DIR="${2}"
CODE_DIR="${3}"

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
python -u ${CODE_DIR}/generate_obs.py ${PRIOR_DIR} ${OUTPUT_DIR} ${CODE_DIR}
