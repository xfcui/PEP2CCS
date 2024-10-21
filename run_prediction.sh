#!/bin/bash

# Check if the user provided a data path
if [ -z "$1" ]; then
  echo "Error: You need to provide a data path!"
  echo "Usage: ./run_prediction.sh /path/to/test.csv"
  exit 1
fi

# Get the data path
DATA_PATH=$1

# Check if the file exists
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: The file '$DATA_PATH' does not exist!"
  exit 1
fi

# Print confirmation message
echo "Using data path '$DATA_PATH' for prediction..."

# Call the Python script for prediction
python3 /root/PEP2CCS/src/predict.py --data_path "$DATA_PATH"

# Check if the prediction was successful
if [ $? -eq 0 ]; then
  echo "Prediction completed! Results have been saved."
else
  echo "Error: An error occurred during prediction."
fi
