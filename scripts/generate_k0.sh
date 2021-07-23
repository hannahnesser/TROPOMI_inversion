#!/bin/bash

jid1=$(sbatch generate_k0_nstate.sh)

sbatch --dependency=afterok:${jid1##* } generate_k0_monthly.sh

sbatch generate_k0_monthly.sh
