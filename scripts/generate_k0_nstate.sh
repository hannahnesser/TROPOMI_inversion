#!/bin/bash

#SBATCH -J save_k0_nstate
#SBATCH -o %x_%j_%a.out
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 30000
#SBATCH -t 0-05:00
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
DATA_DIR="${1}"
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

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/generate_k0_nstate.py ${DATA_DIR} ${CODE_DIR}

# And make a symbolic link
ln -sf "${DATA_DIR}k0_nstate.nc" "${OUTPUT_DIR}k0_nstate.nc"
