<p align="right"><a href="README.md">中文</a> | <b>English</b></p>

# PEP2CCS

PEP2CCS predicts peptide collision cross section (CCS) from sequence, charge, length, and m/z. The network learns a residual on top of the log of the physical baseline `a * charge * sqrt(m/z) + b`.

**Paper:** [How to Integrate Non-Sequence Features for Peptide Collision Cross Section Prediction with Deep Learning?](https://ieeexplore.ieee.org/abstract/document/10822507)  
IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2024. DOI: [10.1109/BIBM62325.2024.10822507](https://doi.org/10.1109/BIBM62325.2024.10822507)

Authors: Zhimeng Tian, Zizheng Nie, Yong Zhang, Daming Zhu, Xuefeng Cui  
Contact: xfcui@email.sdu.edu.cn

## Setup

Conda and Python 3.9 are required. Package versions are pinned in `requirements.txt` and `environment.yml`.

```bash
bash run_setup.sh
conda activate PEP2CCS
```

## Data

The example table is `data/examples/test_data.zip`. The first prediction or experiment run extracts it to `data/examples/test_data.csv`.

Prediction requires `Sequence`, `Charge`, and `m/z`. `Length` is optional and falls back to the sequence length. Evaluation and training also require `CCS`. Supported charges are 2, 3, and 4. Sequences use the 20 standard amino acids and, with the start, charge, and end tokens, must fit in 64 positions.

Column definitions, the two vocabularies, and the physical coefficients are in the [data schema](docs/data_schema.en.md). Notes on the example table and the weights are in [data/README.en.md](data/README.en.md) and [checkpoints/README.en.md](checkpoints/README.en.md).

## Prediction

```bash
bash run_prediction.sh data/examples/test_data.csv
```

If you omit the path, the script uses the example table. Results are written to `results/predict.csv`. When the table contains `CCS`, the terminal prints MSE and the median absolute relative deviation. Relative deviation is `(CCS / Pred CCS) * 100 - 100`.

```bash
python scripts/predict.py --data /path/to/peptides.csv --output results/mine.csv --device auto
```

The device is `cuda` (the default visible GPU) or CPU.

## Reproducing the experiments

```bash
python scripts/run_experiment.py exp1
python scripts/run_experiment.py exp2
python scripts/run_experiment.py exp3
python scripts/run_experiment.py exp4
python scripts/run_experiment.py exp5
```

Model, checkpoint, and data settings for each experiment are in `configs/experiments/`. Metrics are written to `results/<experiment>/metrics.csv`. Check the pipeline on the first 128 rows:

```bash
python scripts/run_experiment.py exp1 --limit 128 --device cpu
```

`--limit` only checks that the run starts. Numbers in the paper come from the full example table. The Exp1 figure annotates the median absolute relative deviation computed from that run.

The mapping from experiments to checkpoints is in the [experiment notes](docs/experiments.en.md).

## Training

`scripts/train.py` rebuilds a training loop from `configs/model.yaml`: Adam, a 10× learning-rate drop at epoch 200, a 1% validation split, split seed 1115, MSE in log CCS, and a stop after 15 epochs without a validation improvement. A newly trained file can be used for prediction. `checkpoints/pep2ccs.pt` remains the checkpoint released with the paper.

```bash
python scripts/train.py --train-csv /path/to/train.csv --output checkpoints/rerun.pt
```

The training CSV uses the same columns as the example table and must include `CCS`. The released checkpoints are ready for prediction and for the experiments above. The `.pt` files are packed in zip archives under `checkpoints/archives/` and are extracted into `checkpoints/` the first time they are used.

## Layout

```text
configs/          hyperparameters, physical coefficients, five experiments
data/examples/    example test table
checkpoints/      released weights
src/pep2ccs/      data, model, training, and evaluation
scripts/          predict, train, and reproduce experiments
tests/            encoding, forward shapes, and checkpoint compatibility
docs/             data schema and experiment notes. English pages use the .en.md suffix
```

## Tests

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
