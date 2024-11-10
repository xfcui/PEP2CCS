#!/bin/bash

echo "##Exp5: The role of Mix Pooling"
python ./src/Exp5/test.py --model_choice 1 --model_path "./src/Exp5/model1.pt"
python ./src/Exp5/test.py --model_choice 2 --model_path "./src/Exp5/model2.pt"
python ./src/Exp5/test.py --model_choice 3 --model_path "./src/Exp5/model3.pt"
python ./src/Exp5/test.py --model_choice 4 --model_path "./src/Exp5/model4.pt"
python ./src/Exp5/test.py --model_choice 5 --model_path "./src/Exp5/model5.pt"
python ./src/Exp5/test.py --model_choice 6 --model_path "./src/Exp5/model6.pt"
python ./src/Exp5/test.py --model_choice 7 --model_path "./src/Exp5/model7.pt"

echo "##Exp 5 replication completed!"
