#!/bin/bash

echo "##Exp2: The function of charge state"
python /root/PEP2CCS/src/Exp2/test1.py /root/PEP2CCS/checkpoint/model.pt 1
python /root/PEP2CCS/src/Exp2/test1.py /root/PEP2CCS/src/Exp2/PEP2CCS.pt 1
python /root/PEP2CCS/src/Exp2/test2.py /root/PEP2CCS/src/Exp2/LSTM.pt 1


python /root/PEP2CCS/src/Exp2/test1.py /root/PEP2CCS/checkpoint/model.pt 0
python /root/PEP2CCS/src/Exp2/test1.py /root/PEP2CCS/src/Exp2/PEP2CCS.pt 0
python /root/PEP2CCS/src/Exp2/test2.py /root/PEP2CCS/src/Exp2/LSTM.pt 0

echo "##Exp 2 replication completed!"
