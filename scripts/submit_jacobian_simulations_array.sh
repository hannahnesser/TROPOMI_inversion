#!/bin/bash

sbatch --array={START}-{END} run_jacobian_simulations.sh

exit 0
