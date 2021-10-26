#!/bin/bash

# Set variables
CODE_PATH=codepathcodepath
SCRIPT_PATH=scriptpathscriptpath
JAC_PATH=$(pwd -P)

# Update code and scripts
cp ${CODE_PATH}/*.py ${JAC_PATH}/python
cp ${SCRIPT_PATH}/*.sh ${JAC_PATH}/scripts

# Submit jobs
sbatch --array={START}-{END} -W run_jacobian_simulations.sh

exit 0
