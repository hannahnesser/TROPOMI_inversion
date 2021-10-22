#!/bin/bash

#SBATCH -J clean_up
#SBATCH -o %x_%j_%a.out
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 2000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END

GC_CH4_DIR=${1}

rm HEMCO_restart.*
rm ${GC_CH4_DIR}/*
