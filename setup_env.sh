#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate caduceus
 elif [[ "${CONDA_PREFIX}" != *"/caduceus" ]]; then
  conda deactivate
  conda activate caduceus
fi

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}"
