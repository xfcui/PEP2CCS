#!/bin/bash

echo "##Exp5: The function of Mix Pooling"
python /root/PEP2CCS/src/Exp5/test.py 1 /root/PEP2CCS/checkpoint/model1.pt
python /root/PEP2CCS/src/Exp5/test.py 2 /root/PEP2CCS/checkpoint/model2.pt
python /root/PEP2CCS/src/Exp5/test.py 3 /root/PEP2CCS/checkpoint/model3.pt
python /root/PEP2CCS/src/Exp5/test.py 4 /root/PEP2CCS/checkpoint/model4.pt
python /root/PEP2CCS/src/Exp5/test.py 5 /root/PEP2CCS/checkpoint/model5.pt
python /root/PEP2CCS/src/Exp5/test.py 6 /root/PEP2CCS/checkpoint/model6.pt
python /root/PEP2CCS/src/Exp5/test.py 7 /root/PEP2CCS/checkpoint/model7.pt

echo "##Exp 5 replication completed!"
