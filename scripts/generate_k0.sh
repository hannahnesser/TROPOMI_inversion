#!/bin/bash

jid1=$(sbatch generate_k0_nstate.sh)

sbatch --dependency=afterok:${jid1##* } --array=1-12 generate_k0_monthly.sh

sbatch --array=1-12 generate_k0_monthly.sh
sbatch --array=2 generate_k0_monthly.sh
