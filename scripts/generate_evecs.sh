#!/bin/bash

#SBATCH -J save_evecs
#SBATCH -o %x_%j.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-02:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

## -------------------------------------------------------------------------##
## Load and prepare the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

## -------------------------------------------------------------------------##
## Initial up
## -------------------------------------------------------------------------##
DATA_DIR=${3}
SUFFIX=${10}
OPT_BC=${5}
if [[ ${OPT_BC} == "True" ]]; then
  SUFFIX="_bc${SUFFIX}"
fi

rm -rf ${DATA_DIR}/evecs_dask_worker${SUFFIX}/

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/generate_evecs.py ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}/evecs_dask_worker${SUFFIX}/
