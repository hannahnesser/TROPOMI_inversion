#!/bin/bash

#SBATCH -J solve_inv
#SBATCH -o %x_%j_%a.out
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-05:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

## Eventually change to 12 cores, 45000 mem, and at least one hour. Just debugging
## for now

## -------------------------------------------------------------------------##
## Load and prepare the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

DATA_DIR=${2}
rm -rf ${DATA_DIR}/dask-worker-space*

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/solve_inversion.py ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}/dask-worker-space*