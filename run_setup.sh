#!/bin/bash

ENV_NAME="PEP2CCS"

conda create -y -n $ENV_NAME python=3.9

source activate $ENV_NAME

pip install -r requirements.txt

echo "Installation is complete! The following libraries are installed in the environment $ENV_NAME:"
pip list

unzip ./src/data/test_data.zip -d ./src/data/
