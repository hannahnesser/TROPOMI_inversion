#!/bin/bash
# shellcheck shell=sh
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

if [ -n "${ZSH_VERSION:-}" ] ; then     # only executed in zsh
    SCRIPTPATH="${0:A:h}"
elif [ -n "${KSH_VERSION:-}" ] ; then   # only executed in ksh or mksh or lksh
    if [ "$(set | grep KSH_VERSION)" = "KSH_VERSION=.sh.version" ] ; then # ksh
      if [ "$(cd "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")" \
        != "$(cd "$(dirname -- "${.sh.file}")" && pwd -P)/$(basename -- "${.sh.file}")" ] ; then
        FILEPATH="${.sh.file}" || usage "$0" ;
      fi
    else # mksh or lksh detected (also check for [lm]ksh renamed as ksh)
      _lmksh="$(basename -- "$0")" ;
      if [ "mksh" = "$_lmksh" ] || [ "lksh" = "$_lmksh" ] || [ "ksh" = "$_lmksh" ] ; then
        # force [lm]ksh to issue error msg; contains this script's rel/path/filename
        FILEPATH="$( (echo "${.sh.file}") 2>&1 )" || : ;
        FILEPATH="$(expr "$vars_script_name" : '^.*ksh: \(.*\)\[[0-9]*\]:')" ;
      fi
    fi
    SCRIPTPATH="${FILEPATH%/*}"
    unset FILEPATH
elif [ -n "${BASH_VERSION:-}" ] ; then  # only executed in bash
    SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
else
    echo "Shell is currently unsupported"
fi
ROOTPATH="${SCRIPTPATH%/*}"
VERSION="$( basename "$ROOTPATH" )"
PARENTDIR="$( dirname "$ROOTPATH" )"
CURRENTENV="$( basename "$PARENTDIR" )"
MAINENV="intelpython"


# Since setvars.sh is being called, only source IntelPython. Code below
# ensures the conda command prompt does not change
if [ -n "$SETVARS_CALL" ]; then
    BACKUP_SETVARS_CALL_PARAMETERS="$@"
    SANITIZED_SETVARS_CALL_PARAMETERS=""
    set -- "$SANITIZED_SETVARS_CALL_PARAMETERS"

    if [[ "$CURRENTENV" == "$MAINENV" ]]; then
        # SAT-3515: only export PS1 if it already existed to more accurately preserve the
        #   user's environment state, improving usability.
        if [ $(env | grep -q PS1) ] ; then
            PS1_IS_SET=1
        else
            PS1_IS_SET=0
        fi

        BACKUP_PROMPT="$PS1"
        . "$ROOTPATH/bin/activate"

        if [ $PS1_IS_SET -eq 1 ] ; then
            export PS1=$BACKUP_PROMPT
        else
            PS1=$BACKUP_PROMPT
        fi
        unset BACKUP_PROMPT
    fi
    set -- "${BACKUP_SETVARS_CALL_PARAMETERS}"
    unset BACKUP_SETVARS_CALL_PARAMETERS
    unset SANITIZED_SETVARS_CALL_PARAMETERS
else
    # Handle instance when SETVARS_CALL=1 is being passed along
    ORIGINAL_PARAMETERS="$@"
    if [ -z "$1" ]; then  # No arguments are supplied
        SANITIZED_PARAMETERS=$ORIGINAL_PARAMETERS
    elif [ "$1" = "SETVARS_CALL=1" ]; then  # One argument is supplied and comes from setvars.sh
        SANITIZED_PARAMETERS=${ORIGINAL_PARAMETERS#"SETVARS_CALL=1"}
    else
        SANITIZED_PARAMETERS=$ORIGINAL_PARAMETERS
    fi

    # The vars.sh script is being sourced within virtual environment
    if [[ "$VERSION" == "latest" ]]; then
        OVERWRITE_CONDA_DEFAULT_ENV="$CURRENTENV"
    else
        OVERWRITE_CONDA_DEFAULT_ENV="$CURRENTENV-$VERSION"
    fi

    set -- "${SANITIZED_PARAMETERS}"
    . "$ROOTPATH/bin/activate"
    unset OVERWRITE_CONDA_DEFAULT_ENV
    set -- "${ORIGINAL_PARAMETERS}"

    unset ORIGINAL_PARAMETERS
    unset SANITIZED_PARAMETERS
fi

unset SCRIPTPATH
unset ROOTPATH
unset VERSION
unset CURRENTENV
unset PARENTDIR
unset MAINENV
