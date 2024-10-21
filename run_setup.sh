#!/bin/bash

echo "Creating conda environment: PEP2CCS"
conda create -n PEP2CCS -y

echo "Activating PEP2CCS environment"
conda activate PEP2CCS

if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Please ensure the file is in the current directory."
    exit 1
fi

echo "Environment setup complete!"
