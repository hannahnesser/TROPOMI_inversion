#!/bin/bash

# Edit apply_TROPOMI_operator.sh to change settings.

sbatch --array=1-12 apply_TROPOMI_operator.sh
