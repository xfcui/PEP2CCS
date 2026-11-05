<p align="right"><a href="README.zh.md">中文</a> | <b>English</b></p>

# PEP2CCS

[![Paper](https://img.shields.io/badge/BIBM%202024-IEEE-004088)](https://ieeexplore.ieee.org/abstract/document/10822507)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FBIBM62325.2024.10822507-blue)](https://doi.org/10.1109/BIBM62325.2024.10822507)
[![Python](https://img.shields.io/badge/python-3.9-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

PEP2CCS predicts peptide collision cross section (CCS) from sequence, charge, length, and m/z. A 6-layer Transformer (embedding size 128, 8 heads) learns a log residual on a charge-specific physical baseline:

```text
CCS = exp( f(sequence, charge, length, m/z) + log(a * charge * sqrt(m/z) + b) )
```

Charge is a special token. It is concatenated with a CNN max pool and passed to the regression head together with length and sqrt(m/z).

[How to Integrate Non-Sequence Features for Peptide Collision Cross Section Prediction with Deep Learning?](https://ieeexplore.ieee.org/abstract/document/10822507), IEEE BIBM 2024. DOI: [10.1109/BIBM62325.2024.10822507](https://doi.org/10.1109/BIBM62325.2024.10822507).  
Zhimeng Tian, Zizheng Nie, Yong Zhang, Daming Zhu, Xuefeng Cui · xfcui@email.sdu.edu.cn

## Installation

Conda and Python 3.9 are required. Dependencies are pinned in `environment.yml`, including PyTorch 2.2.2.

```bash
bash run_setup.sh
conda activate PEP2CCS
```

## Prediction

```bash
bash run_prediction.sh data/examples/test_data.csv
python scripts/predict.py --data /path/to/peptides.csv --output results/mine.csv --device auto
```

Results go to `results/predict.csv`. When `CCS` is present, the terminal reports MSE and the median absolute relative deviation, `(CCS / Pred CCS) * 100 - 100`. Input needs `Sequence`, `Charge`, and `m/z`. Charges are 2, 3, or 4; sequences use the standard amino acids and fit in 64 positions. See the [data schema](docs/data_schema.en.md).

Released weights are attached to [release v1.0.0](https://github.com/xfcui/PEP2CCS/releases/tag/v1.0.0). The first run downloads the archive it needs and extracts it under `checkpoints/`.

## Reproducing the experiments

```bash
python scripts/run_experiment.py exp1   # exp2 … exp5
```

Metrics are written to `results/<experiment>/metrics.csv`. Paper numbers use the full example table. `--limit 128 --device cpu` only checks that a run starts.

| Experiment | Comparison |
| --- | --- |
| Exp1 | Primary model |
| Exp2 | Primary model, a second Transformer checkpoint, and a 3-layer bidirectional LSTM |
| Exp3 | Injecting m/z or length into the charge token |
| Exp4 | Amino-acid composition, an alphabetical vocabulary, and a second set of physical coefficients |
| Exp5 | Seven pooling operators |

Details are in the [experiment notes](docs/experiments.en.md), [data/README.en.md](data/README.en.md), and [checkpoints/README.en.md](checkpoints/README.en.md).

## Training

`scripts/train.py` follows `configs/model.yaml`: Adam, learning rate ×0.1 at epoch 200, a 1% validation split with seed 1115, MSE on log CCS, and a stop after 15 epochs without improvement.

```bash
python scripts/train.py --train-csv /path/to/train.csv --output checkpoints/rerun.pt
```

## Layout

```text
configs/        hyperparameters, physical coefficients, experiment configs
data/examples/  example test table
checkpoints/    extracted weights (downloaded from the release)
src/pep2ccs/    data, model, training, evaluation
scripts/        predict, train, run experiments
docs/           data schema and experiment notes
```

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{tian2024pep2ccs,
  author    = {Tian, Zhimeng and Nie, Zizheng and Zhang, Yong and Zhu, Daming and Cui, Xuefeng},
  title     = {How to Integrate Non-Sequence Features for Peptide Collision Cross Section Prediction with Deep Learning?},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2024},
  doi       = {10.1109/BIBM62325.2024.10822507},
  url       = {https://ieeexplore.ieee.org/abstract/document/10822507}
}
```

[MIT License](LICENSE)
