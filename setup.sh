#!/bin/bash

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (e.g. 3.7) " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. 10.1 or none to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch torchvision cpuonly -c pytorch
else
    conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch -c conda-forge
fi

# install python requirements
pip install -r requirements.txt
