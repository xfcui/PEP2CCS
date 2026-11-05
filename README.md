<p align="right"><b>中文</b> | <a href="README.en.md">English</a></p>

# PEP2CCS

用肽段序列、电荷、长度和 m/z 预测碰撞截面（CCS）。模型在物理基线 `a * charge * sqrt(m/z) + b` 的对数上学习残差。

**论文：** [How to Integrate Non-Sequence Features for Peptide Collision Cross Section Prediction with Deep Learning?](https://ieeexplore.ieee.org/abstract/document/10822507)  
IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2024. DOI: [10.1109/BIBM62325.2024.10822507](https://doi.org/10.1109/BIBM62325.2024.10822507)

作者：Zhimeng Tian，Zizheng Nie，Yong Zhang，Daming Zhu，Xuefeng Cui  
联系：xfcui@email.sdu.edu.cn

## 环境

需要 conda 和 Python 3.9。依赖版本写在 `requirements.txt` 和 `environment.yml`。

```bash
bash run_setup.sh
conda activate PEP2CCS
```

## 数据

示例表在 `data/examples/test_data.zip`，第一次预测或复现实验时会解压为 `data/examples/test_data.csv`。

预测至少需要三列：`Sequence`、`Charge`、`m/z`。`Length` 可以省略，省略时用序列长度。评估和训练还需要 `CCS`。电荷只支持 2、3、4。序列只支持 20 种标准氨基酸，加上起始、电荷和结束标记后不能超过 64。

列定义、两种词表和物理系数见 [数据约定](docs/data_schema.md)。示例表和权重的说明见 [data/README.md](data/README.md) 与 [checkpoints/README.md](checkpoints/README.md)。

## 预测

```bash
bash run_prediction.sh data/examples/test_data.csv
```

未给出路径时使用示例表。结果写到 `results/predict.csv`。表里如果有 `CCS`，终端会打印 MSE 和中位绝对相对偏差。相对偏差的定义是 `(CCS / Pred CCS) * 100 - 100`。

```bash
python scripts/predict.py --data /path/to/peptides.csv --output results/mine.csv --device auto
```

设备使用 `cuda`（当前可见的默认 GPU）或 CPU。

## 复现实验

```bash
python scripts/run_experiment.py exp1
python scripts/run_experiment.py exp2
python scripts/run_experiment.py exp3
python scripts/run_experiment.py exp4
python scripts/run_experiment.py exp5
```

每个实验的模型、权重和数据约定在 `configs/experiments/`。指标写到 `results/<实验名>/metrics.csv`。先确认流程可以用前 128 行：

```bash
python scripts/run_experiment.py exp1 --limit 128 --device cpu
```

`--limit` 只用于检查能否跑通。论文中的测试集结果需要在完整示例表上运行。Exp1 图上的中位绝对相对偏差由当次预测计算。

实验和权重的对应关系见 [实验说明](docs/experiments.md)。

## 训练

`scripts/train.py` 按 `configs/model.yaml` 里的超参重建训练循环：Adam、第 200 个 epoch 学习率乘 0.1、验证集比例 0.01、划分种子 1115、在 log CCS 上做 MSE、15 个 epoch 验证损失不下降则停止。新训出的权重可以用于预测；发布的 `checkpoints/pep2ccs.pt` 仍是论文对应的权重。

```bash
python scripts/train.py --train-csv /path/to/train.csv --output checkpoints/rerun.pt
```

训练集需要与示例表相同的列，并且包含 `CCS`。发布的权重可以直接用于预测和实验复现。`.pt` 文件放在 `checkpoints/archives/` 的 zip 里，第一次使用时解压到 `checkpoints/` 下对应的路径。

## 目录

```text
configs/          超参、物理系数、五个实验
data/examples/    示例测试表
checkpoints/      已发布权重
src/pep2ccs/      数据、模型、训练和评测
scripts/          预测、训练、复现实验
tests/            编码、前向形状、权重是否对得上结构
docs/             数据约定和实验说明。英文页面文件名以 .en.md 结尾
```

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 引用

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
