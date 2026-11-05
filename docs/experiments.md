<p align="right"><a href="../README.md">README</a> · <b>中文</b> | <a href="experiments.en.md">English</a></p>

# 实验

所有实验都读同一张肽段表，差异在模型结构和预处理。权重里的参数名与原来的 `model_param` 一致。

| 实验 | 模型名 | 权重 | 和主模型的差别 |
|---|---|---|---|
| Exp1 | pep2ccs | checkpoints/pep2ccs.pt | 电荷 token 与 CNN max pooling 拼接，回归头再接长度和 sqrt(m/z) |
| Exp2 | pep2ccs | checkpoints/pep2ccs.pt | 与 Exp1 相同的主模型 |
| Exp2 | exp2_pep2ccs | checkpoints/exp2/pep2ccs.pt | 词表行数是 50，回归输入是 259，头里没有 dropout。见下文 |
| Exp2 | lstm | checkpoints/exp2/lstm.pt | 3 层双向 LSTM，直接预测 log CCS。`model.yaml` 里的 6 层是 Transformer 的深度 |
| Exp3 | exp3_mz_embed | checkpoints/exp3/model1.pt | sqrt(m/z) 和电荷写入电荷 token，回归头只用长度 |
| Exp3 | exp3_length_embed | checkpoints/exp3/model2.pt | 长度和电荷写入电荷 token，回归头只用 sqrt(m/z) |
| Exp3 | exp3_joint_embed | checkpoints/exp3/model3.pt | 电荷、长度、m/z 都写入电荷 token |
| Exp4 | exp4 | checkpoints/exp4/model.pt | 字母序词表、另一套物理系数、长度加 3、组成向量。词表和系数见 [数据约定](data_schema.md) |
| Exp5 | exp5_cls … exp5_mean_cnn | checkpoints/exp5/model1.pt … model7.pt | 只改 pooling，见下表 |

Exp2 会各跑一遍全表和“同一序列出现过多种电荷”的子集。

`checkpoints/exp2/pep2ccs.pt` 对不上原来的 `src/model/model.py`。那份代码的词嵌入是 32 行、回归输入是 258，并且头里有 dropout；这个权重的词嵌入是 50 行、回归输入是 259，头是三层全连接加 LayerNorm，没有 dropout。原来的 `test1.py` 会在 `load_state_dict` 时报尺寸不一致。`exp2_pep2ccs` 按权重里的参数名搭建。pooling 仍是电荷 token 加 CNN max。回归头在主模型的长度和 `sqrt(m/z)` 之后再接上数值电荷，输出加上 `log(baseline)`。标准词表的 id 都小于 26，落在 50 行嵌入里。这个文件的前向在原仓库里没有留下，三种标量的排列在 512 行探针上的中位绝对相对偏差都在约 3% 到 5%，不要把它当成主权重的另一份拷贝。

## Exp5 pooling

| 权重 | 模型名 | pooling |
|---|---|---|
| model1.pt | exp5_cls | 只保留电荷 token。卷积层仍在，和原结构的参数名一致 |
| model2.pt | exp5_cls_max | 电荷 token 与屏蔽填充后的 max pooling |
| model3.pt | exp5_max | 只有 max pooling |
| model4.pt | exp5_cnn_max | CNN 之后做 max pooling |
| model5.pt | exp5_mean | 按长度做平均 |
| model6.pt | exp5_mean_max | 平均与 max pooling 拼接 |
| model7.pt | exp5_mean_cnn | 平均与 CNN max pooling 拼接 |

平均 pooling 的长度张量与激活放在同一设备上，因此 CPU 和单卡机器都可以跑。权重文件说明见 [checkpoints/README.md](../checkpoints/README.md)。

## 训练循环

`epochs`、`learning_rate`、`reduce_lr_step`、`early_stop`、`eval_interval` 和划分种子 1115 来自原来的 `model_params.py`。原来的优化代码不在仓库里，所以 `src/pep2ccs/engine/train.py` 是按这些超参重建的：

- 优化器是 Adam
- `StepLR` 的 step size 为 200，gamma 为 0.1
- 损失是预测值与 `log(CCS)` 的 MSE。网络输出已经在对数空间
- 每个 epoch 都在验证集上检查，连续 15 个 epoch 没有下降就停止
- 每 `eval_interval` 个 epoch 打印一次
- 保存验证损失最好的 `{model_param: state_dict}`

发布权重的产生环境里没有记录 PyTorch 版本。加载测试针对 PyTorch 2.2。
