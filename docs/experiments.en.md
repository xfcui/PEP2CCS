<p align="right"><a href="../README.en.md">README</a> · <a href="experiments.md">中文</a> | <b>English</b></p>

# Experiments

Every experiment reads the same peptide table. They differ in architecture and preprocessing. Parameter names inside the weights match the original `model_param` entry.

| Experiment | Model name | Checkpoint | Difference from the main model |
|---|---|---|---|
| Exp1 | pep2ccs | checkpoints/pep2ccs.pt | Charge token concatenated with CNN max pooling; the regression head then receives length and sqrt(m/z) |
| Exp2 | pep2ccs | checkpoints/pep2ccs.pt | Same main model as Exp1 |
| Exp2 | exp2_pep2ccs | checkpoints/exp2/pep2ccs.pt | Vocabulary has 50 rows and the regression input is 259, with no dropout in the head. See below |
| Exp2 | lstm | checkpoints/exp2/lstm.pt | 3-layer bidirectional LSTM that predicts log CCS directly. The 6 layers in `model.yaml` are the Transformer depth |
| Exp3 | exp3_mz_embed | checkpoints/exp3/model1.pt | sqrt(m/z) and charge are written into the charge token; the head uses length only |
| Exp3 | exp3_length_embed | checkpoints/exp3/model2.pt | Length and charge are written into the charge token; the head uses sqrt(m/z) only |
| Exp3 | exp3_joint_embed | checkpoints/exp3/model3.pt | Charge, length, and m/z are all written into the charge token |
| Exp4 | exp4 | checkpoints/exp4/model.pt | Alphabetical vocabulary, a separate physical baseline, length plus 3, and a composition vector. Vocabulary and coefficients are in the [data schema](data_schema.en.md) |
| Exp5 | exp5_cls … exp5_mean_cnn | checkpoints/exp5/model1.pt … model7.pt | Pooling only; see the table below |

Exp2 is run once on the full table and once on sequences observed at more than one charge.

`checkpoints/exp2/pep2ccs.pt` does not match the old `src/model/model.py`. That code used a 32-row embedding, a regression input of 258, and dropout in the head. This file has a 50-row embedding, a regression input of 259, and a three-layer head with LayerNorm and no dropout. The old `test1.py` fails in `load_state_dict` because the sizes differ. `exp2_pep2ccs` follows the parameter names stored in the file. Pooling is still the charge token plus CNN max. The head receives length and `sqrt(m/z)`, as in the main model, and then the numeric charge. The output is added to `log(baseline)`. Standard vocabulary ids are below 26, so they fit in the 50-row embedding. The original repository did not include the forward pass for this file. On a 512-row probe, every order of the three scalars gives a median absolute relative deviation of about 3% to 5%. Treat this file as its own checkpoint, separate from the main weights.

## Exp5 pooling

| Checkpoint | Model name | Pooling |
|---|---|---|
| model1.pt | exp5_cls | Charge token only. The convolution is still present so the parameter names match the original module |
| model2.pt | exp5_cls_max | Charge token concatenated with max pooling after padding is masked |
| model3.pt | exp5_max | Max pooling only |
| model4.pt | exp5_cnn_max | Max pooling after the CNN |
| model5.pt | exp5_mean | Mean over the length |
| model6.pt | exp5_mean_max | Mean concatenated with max pooling |
| model7.pt | exp5_mean_cnn | Mean concatenated with CNN max pooling |

Mean pooling keeps the length tensor on the same device as the activations, so the run works on CPU and on a single GPU. Checkpoint notes are in [checkpoints/README.en.md](../checkpoints/README.en.md).

## Training loop

`epochs`, `learning_rate`, `reduce_lr_step`, `early_stop`, `eval_interval`, and split seed 1115 come from the original `model_params.py`. The original optimization code was not in the repository, so `src/pep2ccs/engine/train.py` follows those hyperparameters:

- The optimizer is Adam.
- `StepLR` uses step size 200 and gamma 0.1.
- The loss is MSE between the prediction and `log(CCS)`. The network already emits a log-space value.
- Every epoch is checked on the validation split. Training stops after 15 epochs without an improvement.
- A line is printed every `eval_interval` epochs.
- The best validation loss is saved as `{model_param: state_dict}`.

The environment that produced the released weights did not record a PyTorch version. The load tests target PyTorch 2.2.
