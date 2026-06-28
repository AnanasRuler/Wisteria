#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate wisteria
 elif [[ "${CONDA_PREFIX}" != *"/wisteria" ]]; then
  conda deactivate
  conda activate wisteria
fi

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}"
