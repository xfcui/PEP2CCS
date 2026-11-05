<p align="right"><a href="../README.zh.md">README</a> · <b>中文</b> | <a href="data_schema.en.md">English</a></p>

# 数据约定

示例文件 `data/examples/test_data.zip` 解压后约 21 万行。预测只用下面这些列，其余列会原样留在输出里。

| 列 | 是否必需 | 含义 |
|---|---|---|
| Sequence | 是 | 未修饰的肽段序列，只含 20 种标准氨基酸 |
| Charge | 是 | 2、3 或 4 |
| m/z | 是 | 质荷比，必须为正 |
| Length | 否 | 氨基酸个数。缺省时用 `len(Sequence)` |
| CCS | 评估和训练需要 | 观测碰撞截面，必须为正 |

`Modified_sequence` 只作记录。编码使用去掉修饰标记后的 `Sequence`。

## 词表

主模型和 Exp1、Exp2、Exp3、Exp5 使用 `standard` 顺序 `ARNDCQEGHILKMFPSTWYV`。

| token | id |
|---|---|
| PAD | 0 |
| SOS | 1 |
| EOS | 2 |
| Charge2 / Charge3 / Charge4 | 3 / 4 / 5 |
| 氨基酸 | 从 6 起，按上面的顺序 |

一条样本编码为 `[SOS, Charge, 氨基酸..., EOS, PAD...]`，总长 64。因此肽段最长 61。未知字符、其他电荷和超长序列会直接报错。

Exp4 的 token 顺序是 20 种氨基酸的字母序，因为原脚本用 `sorted` 从数据里建词表。示例表包含全部 20 种氨基酸，所以这个固定词表和“按测试表排序”得到的 id 一致。Exp4 另外把长度加 3，并附上 20 维残基计数和 3 维电荷 one-hot。

## 物理基线

`configs/physics.yaml`：

`baseline = a * charge * sqrt(m/z) + b`

网络输出加上 `log(baseline)`，再取指数得到 CCS。Exp4 使用文件里的 `exp4` 系数，其他实验使用 `default`。各实验如何使用这些系数见 [实验说明](experiments.md)。
