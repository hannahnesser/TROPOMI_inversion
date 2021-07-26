#!/bin/bash

#SBATCH -J save_pph0
#SBATCH -o %x_%j_%a.out
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 1000
#SBATCH -t 0-10:00
#SBATCH --mail-type=END

## Eventually change to 12 cores, 45000 mem, and at least one hour. Just debugging
## for now

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
DATA_DIR="${1}"
CODE_DIR="${2}"
MONTH="${SLURM_ARRAY_TASK_ID}"
MEMORY_GB=45

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
python -u ${python_dir}/python/generate_pph0.py ${MONTH} ${MEMORY_GB} ${DATA_DIR} ${CODE_DIR}
