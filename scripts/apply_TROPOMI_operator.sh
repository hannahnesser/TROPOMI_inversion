#!/bin/bash

# SBATCH -J TROPOMI_operator
# SBATCH -c 2
# SBATCH -N 1
# SBATCH -p huce_intel
# SBATCH --mem 7000
# SBATCH -t 0-02:00

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
TROPOMI_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
GC_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/OutputDir/"
OUTPUT_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir/"

# Latitude and longitude range
LONS="-130 -60 0.25"
LATS="9.75 60 0.3125"
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
## Run the script
## -------------------------------------------------------------------------##
python_dir=$(dirname `pwd`)
cd $GC_DATA_DIR
mkdir -p $OUTPUT_DIR
python ${python_dir}/python/GC_to_TROPOMI.py $TROPOMI_DATA_DIR $GC_DATA_DIR $OUTPUT_DIR $LONS $LATS $BUFFER $YEAR $MONTH
