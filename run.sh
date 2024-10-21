#!/bin/bash

echo "##Exp3: The function of m/z and length"
python /root/PEP2CCS/src/Exp3/test.py /root/PEP2CCS/checkpoint/model1.pt
python /root/PEP2CCS/src/Exp3/test.py /root/PEP2CCS/checkpoint/model2.pt
python /root/PEP2CCS/src/Exp3/test.py /root/PEP2CCS/checkpoint/model3.pt
python /root/PEP2CCS/src/Exp3/test.py

echo "##Exp 3 replication completed!"
