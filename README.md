# Exp3: The function of m/z and length

## Introduction
---
| **Index** | **Before Embedding** |              | **After Pooling**  |              |              |
|:---------:|:-------------------:|:------------:|:-----------------:|:------------:|:------------:|
|           | **m/z**              | **length**   | **m/z**           | **length**   | **MAPE(%)**  |
|:---------:|:-------------------:|:------------:|:-----------------:|:------------:|:------------:|
| model1         | ✓                   | --           | --                | ✓            | 1.146%       |
| model2         | --                  | ✓            | ✓                 | --           | 1.144%       |
| model3         | ✓                   | ✓            | --                | --           | 1.165%       |
| **PEP2CCS**         | --                  | --           | ✓                 | ✓            | **1.139%**   |

**Note:** Before Embedding refers to the first dashed box in Fig 1 and After Pooling refers to the second dashed box in Fig 1.


