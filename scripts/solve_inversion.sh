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
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_correct.nc"
SA_FILE="${SHORT_TERM_DATA_DIR}/sa.nc"

# Inversion files: observations (DEFAULT)
YA_FILE="${SHORT_TERM_DATA_DIR}/ya.nc" # If niter = 0, this should be ya0
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t.nc" # Regridded errors (2x2) with threshold

# Scaling factors
SA_SCALE="1"
RF="1"
PCT_OF_INFO="80"
EVEC_SCALE_FACTOR="1.0E-8"

# Saving out ("None" if original)
FILE_SUFFIX="_rg2rt_10t"

## Sensitivity inversion options #

#

# # -------------------------------------------------------------#
# # Variable errors
# SA_FILE="${SHORT_TERM_DATA_DIR}/sa_var_max.nc"
# FILE_SUFFIX="_rg2rt_10t_sa_var_max"
# # -------------------------------------------------------------#

# -------------------------------------------------------------#
# Boundary condition test (Running with both sa standard and sa_var_max)
OPTIMIZE_BC="True"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_bc0.nc"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Permian sensitivity test
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_edf.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_edf.nc"
FILE_SUFFIX="_rg2rt_10t_edf"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Permian sensitivity test + NLC
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_edf.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_edf_nlc.nc"
FILE_SUFFIX="_rg2rt_10t_edf_nlc"
# -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Wetland 50% test
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w50.nc"
# FILE_SUFFIX="_rg2rt_10t_w50"
# # -------------------------------------------------------------#

# -------------------------------------------------------------#
# Wetland 4.04 scaling test
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w404.nc"
FILE_SUFFIX="_rg2rt_10t_w404"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Wetland 4.04 scaling test + NLC
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w404_nlc.nc"
FILE_SUFFIX="_rg2rt_10t_w404_nlc"
# -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Wetland remove ensemble members 3 (1923) and 7 (2913) test
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37.nc"
# FILE_SUFFIX="_rg2rt_10t_w37"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine BC0 and wetland 4.04 scaling test
# OPTIMIZE_BC="True"
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_bc0.nc"
# FILE_SUFFIX="_rg2rt_10t_w404_bc0"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine BC0 and EDF test
# OPTIMIZE_BC="True"
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_edf_bc0.nc"
# FILE_SUFFIX="_rg2rt_10t_edf_bc0"
# # -------------------------------------------------------------#

# -------------------------------------------------------------#
# Combine wetlands 4.04 and EDF test
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_edf.nc"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w404_edf.nc"
FILE_SUFFIX="_rg2rt_10t_w404_edf"
# -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine wetlands 37 and EDF test
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w37_edf.nc"
# FILE_SUFFIX="_rg2rt_10t_w37_edf"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine wetlands 50 and EDF test
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w50_edf.nc"
# FILE_SUFFIX="_rg2rt_10t_w50_edf"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine wetlands 4.04 and EDF test and NLC
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_edf.nc"
# SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_nlc.nc" # Regridded errors (2x2) with threshold
# YA_FILE="${SHORT_TERM_DATA_DIR}/ya_nlc.nc" # If niter = 0, this should be ya0
# FILE_SUFFIX="_rg2rt_10t_w404_edf_nlc"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine wetlands 4.04 and NLC
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404.nc"
# SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_nlc.nc" # Regridded errors (2x2) with threshold
# YA_FILE="${SHORT_TERM_DATA_DIR}/ya_nlc.nc" # If niter = 0, this should be ya0
# FILE_SUFFIX="_rg2rt_10t_w404_nlc"
# # -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Combine BC0 and wetlands 4.04 and NLC
# OPTIMIZE_BC="True"
# XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_bc0.nc"
# SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_nlc.nc" # Regridded errors (2x2) with threshold
# YA_FILE="${SHORT_TERM_DATA_DIR}/ya_nlc.nc" # If niter = 0, this should be ya0
# FILE_SUFFIX="_rg2rt_10t_w404_nlc_bc0"
# # -------------------------------------------------------------#

# -------------------------------------------------------------#
# Combine BC0 and wetlands 4.04 and EDF test and NLC
OPTIMIZE_BC="True"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_edf_bc0.nc"
SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg2rt_10t_nlc.nc" # Regridded errors (2x2) with threshold
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_nlc.nc" # If niter = 0, this should be ya0
FILE_SUFFIX="_rg2rt_10t_w404_edf_nlc_bc0"
# -------------------------------------------------------------#

# -------------------------------------------------------------#
# Combine BC0, EDF, and wetland 4.04 scaling test
OPTIMIZE_BC="True"
YA_FILE="${SHORT_TERM_DATA_DIR}/ya_w404_edf.nc"
XA_ABS_FILE="${SHORT_TERM_DATA_DIR}/xa_abs_w404_edf_bc0.nc"
FILE_SUFFIX="_rg2rt_10t_w404_edf_bc0"
# -------------------------------------------------------------#

# # -------------------------------------------------------------#
# # Blended albedo threshold of 0.85
# SO_FILE="${SHORT_TERM_DATA_DIR}/so_rg4rt.nc"
# YA_FILE="${SHORT_TERM_DATA_DIR}/ya_ba85.nc"
# FILE_SUFFIX="_rg4rt_ba85"
# # -------------------------------------------------------------#

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
jid4=$(sbatch --dependency=afterok:${jid3##* } run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${XA_ABS_FILE} ${SA_FILE} ${SA_SCALE} ${SO_FILE} ${RF} ${YA_FILE} ${PCT_OF_INFO} ${EVEC_SCALE_FACTOR} ${FILE_SUFFIX} ${CODE_DIR})
# jid4=$(sbatch run_solve_inversion.sh "2" ${SHORT_TERM_DATA_DIR} ${OPTIMIZE_BC} ${OPTIMIZE_RF} ${XA_ABS_FILE} ${SA_FILE} ${SA_SCALE} ${SO_FILE} ${RF} ${YA_FILE} ${PCT_OF_INFO} ${EVEC_SCALE_FACTOR} ${FILE_SUFFIX} ${CODE_DIR})
