#!/bin/bash

#SBATCH -J save_pph0
#SBATCH -o %x_%j_%a.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
DATA_DIR="${1}"
CODE_DIR="${2}"
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

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/generate_pph0.py ${MONTH} ${DATA_DIR} ${CODE_DIR}