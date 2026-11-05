<p align="right"><b>中文</b> | <a href="README.md">English</a></p>

# PEP2CCS

[![Paper](https://img.shields.io/badge/BIBM%202024-IEEE-004088)](https://ieeexplore.ieee.org/abstract/document/10822507)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FBIBM62325.2024.10822507-blue)](https://doi.org/10.1109/BIBM62325.2024.10822507)
[![Python](https://img.shields.io/badge/python-3.9-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

PEP2CCS 由肽段序列、电荷、长度和 m/z 预测碰撞截面（CCS）。6 层 Transformer（嵌入维度 128，8 个注意力头）在按电荷拟合的物理基线上学习对数残差：

```text
CCS = exp( f(sequence, charge, length, m/z) + log(a · charge · sqrt(m/z) + b) )
```

电荷写成特殊 token，与 CNN max pooling 拼接后，连同长度和 sqrt(m/z) 进入回归头。

[How to Integrate Non-Sequence Features for Peptide Collision Cross Section Prediction with Deep Learning?](https://ieeexplore.ieee.org/abstract/document/10822507), IEEE BIBM 2024. DOI: [10.1109/BIBM62325.2024.10822507](https://doi.org/10.1109/BIBM62325.2024.10822507).  
Zhimeng Tian, Zizheng Nie, Yong Zhang, Daming Zhu, Xuefeng Cui · xfcui@email.sdu.edu.cn

## 安装

需要 conda 与 Python 3.9。依赖锁定在 `environment.yml`（含 PyTorch 2.2.2）。

```bash
bash run_setup.sh
conda activate PEP2CCS
```

## 预测

```bash
bash run_prediction.sh data/examples/test_data.csv
python scripts/predict.py --data /path/to/peptides.csv --output results/mine.csv --device auto
```

结果写入 `results/predict.csv`。表中有 `CCS` 时，终端报告 MSE 和中位绝对相对偏差，定义为 `(CCS / Pred CCS) * 100 - 100`。输入至少包含 `Sequence`、`Charge`、`m/z`；电荷限于 2、3、4，序列为标准氨基酸且总长度不超过 64。列定义见 [数据约定](docs/data_schema.md)。

发布权重附在 [Release v1.0.0](https://github.com/xfcui/PEP2CCS/releases/tag/v1.0.0)。第一次运行会下载用到的压缩包，并解压到 `checkpoints/`。

## 复现实验

```bash
python scripts/run_experiment.py exp1   # exp2 … exp5
```

指标写入 `results/<实验名>/metrics.csv`。论文结果对应完整示例表；`--limit 128 --device cpu` 只检查能否跑通。

| 实验 | 内容 |
| --- | --- |
| Exp1 | 主模型 |
| Exp2 | 主模型、另一份 Transformer 权重、3 层双向 LSTM |
| Exp3 | 把 m/z 或长度写入电荷 token |
| Exp4 | 氨基酸组成、字母序词表、另一组物理系数 |
| Exp5 | 七种 pooling |

细节见 [实验说明](docs/experiments.md)、[data/README.md](data/README.md)、[checkpoints/README.md](checkpoints/README.md)。

## 训练

`scripts/train.py` 按 `configs/model.yaml` 重建训练：Adam，第 200 个 epoch 学习率乘 0.1，种子 1115 划分 1% 验证集，在 log CCS 上做 MSE，连续 15 个 epoch 无改进则停止。

```bash
python scripts/train.py --train-csv /path/to/train.csv --output checkpoints/rerun.pt
```

## 目录

```text
configs/        超参数、物理系数、实验配置
data/examples/  示例测试表
checkpoints/    从 Release 下载后解压的权重
src/pep2ccs/    数据、模型、训练、评测
scripts/        预测、训练、实验
docs/           数据约定与实验说明
```

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

[MIT License](LICENSE)
