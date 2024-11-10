#!/bin/bash

echo "##Exp2: The role of charge state"
python ./src/Exp2/test1.py --model_path "./src/checkpoint/model.pt" --use_full_data 1
python ./src/Exp2/test1.py --model_path "./src/Exp2/PEP2CCS.pt" --use_full_data 1
python ./src/Exp2/test2.py --model_path "./src/Exp2/LSTM.pt" --use_full_data 1


python ./src/Exp2/test1.py --model_path "./src/checkpoint/model.pt" --use_full_data 0
python ./src/Exp2/test1.py --model_path "./src/Exp2/PEP2CCS.pt" --use_full_data 0
python ./src/Exp2/test2.py --model_path "./src/Exp2/LSTM.pt" --use_full_data 0

echo "##Exp 2 replication completed!"
