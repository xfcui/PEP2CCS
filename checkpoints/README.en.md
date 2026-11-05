<p align="right"><a href="../README.en.md">README</a> · <a href="README.md">中文</a> | <b>English</b></p>

# Checkpoints

The repository stores zip archives in `archives/` so the weight directory is not committed as loose files. A `.pt` file is already a zip; packing them again saves a little more, and each archive stays within GitHub's per-file limit. The first load extracts the member listed below into this directory. The files are pickles; load only files you trust. Prediction reads `model_param`.

| Archive | Contents |
|---|---|
| archives/pep2ccs.zip | pep2ccs.pt |
| archives/exp2.zip | exp2/pep2ccs.pt, exp2/lstm.pt |
| archives/exp3.zip | exp3/model1.pt … model3.pt |
| archives/exp4.zip | exp4/model.pt |
| archives/exp5-part1.zip | exp5/model1.pt … model3.pt |
| archives/exp5-part2.zip | exp5/model4.pt … model6.pt |
| archives/exp5-part3.zip | exp5/model7.pt |

| Extracted file | Architecture |
|---|---|
| pep2ccs.pt | Main model, used by Exp1 and Exp2 |
| exp2/pep2ccs.pt | Exp2 PEP2CCS weights. The embedding has 50 rows and the regression input is 259. The structure is in the [experiment notes](../docs/experiments.en.md) |
| exp2/lstm.pt | Exp2 3-layer bidirectional LSTM |
| exp3/model1.pt | m/z written into the embedding |
| exp3/model2.pt | Length written into the embedding |
| exp3/model3.pt | Charge, length, and m/z written into the embedding |
| exp4/model.pt | Composition-vector model |
| exp5/model1.pt … model7.pt | Seven pooling variants, in the same order as the [experiment notes](../docs/experiments.en.md) |

The original repository did not record the PyTorch version that wrote these files. This repository loads them with PyTorch 2.2.2.
