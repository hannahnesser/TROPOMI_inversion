#!/bin/bash

# Edit apply_TROPOMI_operator.sh to change settings.

sbatch --array=1-2 apply_TROPOMI_operator.sh
