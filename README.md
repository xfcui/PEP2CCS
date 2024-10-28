# PEP2CCS
---
The PEP2CCS is a deep learning model designed to predict CCS. This model integrates enhanced physical features of peptides, thereby improving the accuracy of CCS predictions. You can quickly get started with the PEP2CCS according to the following instructions.

```
Authors: Zhimeng Tian, Zizheng Nie, Yong Zhang, Daming Zhu and Xuefeng Cui*
Contact: xfcui@email.sdu.edu.cna
```

## Installation
---
Create virtual environment and install packages:
```bash
chmod +x run_setup.sh
bash ./run_setup.sh
```

## Quick start
---
### Get clone
Clone this repository by:
```
git clone https://github.com/xfcui/PEP2CCS.git
```

### Usage
---
If you want to run our model on your own data, you need to provide the file.

#### Data preparation
After creating a virtual environment, you need to prepare data and trained model. We provide a sample data in the data directory. We also provide the trained model under the checkpoint/model.pt.

### Predict peptide ccs
```bash
chmod +x run_prediction.sh
./run_prediction.sh /path/to/test.csv
```

## Reproduction experiments
---
### Exp1: Performance of PEP2CCS
```bash
chmod +x /root/PEP2CCS/src/Exp1/run.sh
bash /root/PEP2CCS/src/Exp1/run.sh
```

### Exp2: The function of charge state
```bash
chmod +x /root/PEP2CCS/src/Exp2/run.sh
bash /root/PEP2CCS/src/Exp2/run.sh
```

### Exp3: The function of m/z and length
```bash
chmod +x /root/PEP2CCS/src/Exp3/run.sh
bash /root/PEP2CCS/src/Exp3/run.sh
```

### Exp4: Application of m/z
```bash
chmod +x /root/PEP2CCS/src/Exp4/run.sh
bash /root/PEP2CCS/src/Exp4/run.sh
```

### Exp5: The function of Mix Pooling
```bash
chmod +x /root/PEP2CCS/src/Exp5/run.sh
bash /root/PEP2CCS/src/Exp5/run.sh
```

