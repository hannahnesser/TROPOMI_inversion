#!/bin/bash

#SBATCH -J TROPOMI_operator
#SBATCH -o %x_%j_%a.out
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 15000
#SBATCH -t 0-00:40
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
TROPOMI_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
GC_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/OutputDir/"
OUTPUT_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir/"

# Latitude and longitude range
LONS="-130 -60 0.3125"
LATS="9.75 60 0.25"
BUFFER="3 3 3 3"

# time range
YEAR="2019"
MONTH="${SLURM_ARRAY_TASK_ID}"

## -------------------------------------------------------------------------##
## Print out user preferences
## -------------------------------------------------------------------------##
echo "======================================================================="
echo "TROPOMI DATA DIRECTORY:    ${TROPOMI_DATA_DIR}"
echo "GEOS-CHEM DATA DIRECTORY:  ${GC_DATA_DIR}"
echo "OUTPUT DATA DIRECTORY:     ${OUTPUT_DIR}"
echo "LONGITUDE RANGE:           ${LONS}"
echo "LATITUDE RANGE:            ${LATS}"
echo "YEAR:                      ${YEAR}"
echo "MONTH:                     ${MONTH}"
echo "======================================================================="

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
cd $GC_DATA_DIR
mkdir -p $OUTPUT_DIR

echo "Initiating script"
python -u ${python_dir}/python/GC_to_TROPOMI.py $TROPOMI_DATA_DIR $GC_DATA_DIR $OUTPUT_DIR $LONS $LATS $BUFFER $YEAR $MONTH
