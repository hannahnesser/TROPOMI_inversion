#!/bin/bash

#SBATCH -J {RunName}
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 20000
#SBATCH -t 4-00:00

### Run directory
RUNDIR=$(pwd -P)

### Get current task ID
x=${SLURM_ARRAY_TASK_ID}

### Add zeros to the cluster Id
if [ $x -lt 10 ]; then
    xstr="000${x}"
elif [ $x -lt 100 ]; then
    xstr="00${x}"
elif [ $x -lt 1000 ]; then
    xstr="0${x}"
else
    xstr="${x}"
fi

### Run GEOS-Chem in the directory corresponding to the cluster Id
cd  ${RUNDIR}/{RunName}_${xstr}
./{RunName}_${xstr}.run

exit 0
