#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: You need to provide a data path!"
  echo "Usage: ./run_prediction.sh /path/to/test.csv"
  exit 1
fi

DATA_PATH=$1

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: The file '$DATA_PATH' does not exist!"
  exit 1
fi
echo "Using data path '$DATA_PATH' for prediction..."

python ./src/predict/predict.py --data_path "$DATA_PATH"

if [ $? -eq 0 ]; then
  echo "Prediction completed! Results have been saved."
else
  echo "Error: An error occurred during prediction."
fi
