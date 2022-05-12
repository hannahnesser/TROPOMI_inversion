#!/bin/bash

# Directories
PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_NNNN"
NPERT_DIRS=1952
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results"
LONG_TERM_DATA_DIR="/n/seasasfs02/hnesser/TROPOMI_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

# Preferences
NUM_EVECS="1952"
CALCULATE_EVECS="True"
FORMAT_EVECS="False"
OPTIMIZE_RF="False"
OPTIMIZE_BC="False"
CHUNK_SIZE=150000

## Files
# A mask that is True where we want to keep state vector elements
# and false where we want to set the Jacobian columns to 0
MASK="None"

# Inversion files (DEFAULT)
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs.nc"
SA_FILE="${SHORT_TERM_DATA_DIR}/sa.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so.nc" # If niter = 0, this should be so0
YA_FILE="${SHORT_TERM_DATA_DIR}/ya.nc" # If niter = 0, this should be ya0
C_FILE="${SHORT_TERM_DATA_DIR}/c.nc"

# Scaling factors
SA_SCALE="1"
RF="1"

# Saving out
FILE_SUFFIX="None"

## Sensitivity inversion options ##

# -------------------------------------------------------------#
# Regridded errors (2x2)
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg.nc"
FILE_SUFFIX="_rg"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# More coarsly regridded errors (2x2 and monthly or seasonal?)
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rgrt.nc"
FILE_SUFFIX="_rgrt"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# More coarsly regridded errors (3x3 and monthly or seasonal?)
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg3rt.nc"
FILE_SUFFIX="_rg3rt"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# More coarsly regridded errors (4x4 and seasonal)
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg4rt.nc"
FILE_SUFFIX="_rg4rt"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Permian sensitivity test
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg4rt.nc"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_edf.nc"
C_FILE="${SHORT_TERM_DATA_DIR}/c_edf.nc"
FILE_SUFFIX="_rg4rt_edf"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Wetland 50% test
OPTIMIZE_BC="False"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg4rt.nc"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_wetlands50.nc"
C_FILE="${SHORT_TERM_DATA_DIR}/c_wetlands50.nc"
FILE_SUFFIX="_rg4rt_wetlands50"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Boundary condition test
OPTIMIZE_BC="True"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg4rt.nc"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_bc0.nc"
C_FILE="${SHORT_TERM_DATA_DIR}/c_bc0.nc"
FILE_SUFFIX="_rg4rt_bc0"
# -------------------------------------------------------------#


# FILE_SUFFIX="_savar"
# FILE_SUFFIX="_nlc"

# Build the Jacobian
jid1=$(sbatch --array=1-20 build_k_chunks.sh ${CHUNK_SIZE} "2" ${PRIOR_DIR} ${PERT_DIRS} ${NPERT_DIRS} ${SHORT_TERM_DATA_DIR} ${CODE_DIR})

# Calculate the prior preconditioned Hessian
## For this, we assume that the scaling on Sa and on So are 1 (we pass these arguments explicitly
## to avoid confusion)
jid2=$(sbatch --dependency=afterok:${jid1##* } --array=1-20 generate_pph.sh ${CHUNK_SIZE} "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${XA_ABS_FILE} ${SA_FILE} "1" ${SO_FILE} "1" ${YA_FILE} ${FILE_SUFFIX} ${CODE_DIR})
# jid2=$(sbatch --array=1-20 generate_pph.sh ${CHUNK_SIZE} "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${XA_ABS_FILE} ${SA_FILE} "1" ${SO_FILE} "1" ${YA_FILE} ${FILE_SUFFIX} ${CODE_DIR})

# Calculate the eigenvectors
## For this, we assume that the scaling on Sa is 1 (we pass this argument explicitly
## to avoid confusion)
jid3=$(sbatch --dependency=afterok:${jid2##* } generate_evecs.sh "2" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${OPTIMIZE_BC} ${CALCULATE_EVECS} ${FORMAT_EVECS} ${SA_FILE} "1" ${FILE_SUFFIX} ${CODE_DIR})
# jid3=$(sbatch generate_evecs.sh "2" ${NUM_EVECS} ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${OPTIMIZE_BC} ${CALCULATE_EVECS} ${FORMAT_EVECS} ${SA_FILE} "1" ${FILE_SUFFIX} ${CODE_DIR})

# Solve the inversion
jid4=$(sbatch --dependency=afterok:${jid3##* } run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${XA_ABS_FILE} ${SA_FILE} ${SA_SCALE} ${SO_FILE} ${RF} ${YA_FILE} ${C_FILE} ${FILE_SUFFIX} ${CODE_DIR})
# jid4=$(sbatch run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${LONG_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${XA_ABS_FILE} ${SA_FILE} ${SA_SCALE} ${SO_FILE} ${RF} ${YA_FILE} ${C_FILE} ${FILE_SUFFIX} ${CODE_DIR})
