#!/bin/bash

#SBATCH -J solve_inv
#SBATCH -o %x_%j.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-01:00
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
SUFFIX=${12}
OPT_BC=${3}
if [[ ${OPT_BC} == "True" ]]; then
  SUFFIX="_bc${SUFFIX}"
fi
rm -rf ${DATA_DIR}/inv_dask_worker{suffix}/

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/solve_inversion.py ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}/inv_dask_worker{suffix}/
