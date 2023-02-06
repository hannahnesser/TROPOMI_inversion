#!/bin/bash

# Directories
PRIOR_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final"
PERT_DIRS="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_NNNN"
NPERT_DIRS=1952
SHORT_TERM_DATA_DIR="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results"
LONG_TERM_DATA_DIR="/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion"
CODE_DIR="/n/home04/hnesser/TROPOMI_inversion/python"

# Preferences
NUM_EVECS="1952"
CALCULATE_EVECS="True"
FORMAT_EVECS="False"
OPTIMIZE_RF="False"
OPTIMIZE_BC="False"
CHUNK_SIZE=150000

## Files

# Inversion files: priors (DEFAULT)
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs.nc"
SA_FILE="${SHORT_TERM_DATA_DIR}/sa.nc"

# Inversion files: observations (DEFAULT)
YA_FILE="${SHORT_TERM_DATA_DIR}/ya.nc" # If niter = 0, this should be ya0
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t.nc" # Regridded errors (2x2) with threshold

# Inversion files: sectoral breakdown (DEFAULT)
W_FILE="${SHORT_TERM_DATA_DIR}/w.csv"

# Scaling factors
SA_SCALE="1"
RF="1"
PCT_OF_INFO="80"
EVEC_SCALE_FACTOR="10"
DOFS_THRESHOLD="0.05"

# Saving out ("None" if original)
FILE_SUFFIX="_rg2rt_10t"

## ------------------------------------------------------------------ ##
## Sensitivity inversion options
## ------------------------------------------------------------------ ##
# ------------------------------------------------------------- #
# Wetland remove ensemble members 3 and 7 and EDF and BC0 - 
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37_edf_bc0.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w37_edf.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_w37_edf.nc"
W_FILE="${SHORT_TERM_DATA_DIR}/w_w37_edf.csv"
OPTIMIZE_BC="False"
FILE_SUFFIX="_rg2rt_10t_w37_edf_bc0"

# Ensemble member 1
RF="0.175"
SA_SCALE="0.5"

# Ensemble member 2
RF="0.35"
SA_SCALE="0.75"
# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
# Wetland remove ensemble members 3 and 7 and EDF and BC0 and NLC - 
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37_edf_bc0.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w37_edf_nlc.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_w37_edf_nlc.nc"
W_FILE="${SHORT_TERM_DATA_DIR}/w_w37_edf.csv"
OPTIMIZE_BC="False"
FILE_SUFFIX="_rg2rt_10t_w37_edf_bc0_nlc"

# Ensemble member 1
RF="0.175"
SA_SCALE="0.75"
# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
# Wetland remove ensemble members 3 and 7 and EDF and BC0 and BC -
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37_edf_bc0.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w37_edf.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_w37_edf.nc"
W_FILE="${SHORT_TERM_DATA_DIR}/w_w37_edf.csv"
OPTIMIZE_BC="True"
FILE_SUFFIX="_rg2rt_10t_w37_edf_bc0"

# Ensemble member 1
RF="0.2"
SA_SCALE="0.5"

# Ensemble member 2
RF="0.45"
SA_SCALE="0.75"
# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
# Wetland remove ensemble members 3 and 7 and EDF and BC0 and NLC and BC - 
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37_edf_bc0.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w37_edf_nlc.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_w37_edf_nlc.nc"
W_FILE="${SHORT_TERM_DATA_DIR}/w_w37_edf.csv"
OPTIMIZE_BC="True"
FILE_SUFFIX="_rg2rt_10t_w37_edf_bc0_nlc"

# Ensemble member 1
RF="0.175"
SA_SCALE="0.5"

# Ensemble member 2
RF="0.3"
SA_SCALE="0.75"

# Ensemble member 3
RF="0.5"
SA_SCALE="1.0"
# ------------------------------------------------------------- #

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
jid4=$(sbatch --dependency=afterok:${jid3##* } run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${SA_FILE} ${SA_SCALE} ${RF} ${W_FILE} ${PCT_OF_INFO} ${EVEC_SCALE_FACTOR} ${DOFS_THRESHOLD} ${FILE_SUFFIX} ${CODE_DIR})
# jid4=$(sbatch run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${SA_FILE} ${SA_SCALE} ${RF} ${W_FILE} ${PCT_OF_INFO} ${EVEC_SCALE_FACTOR} ${DOFS_THRESHOLD} ${FILE_SUFFIX} ${CODE_DIR})
