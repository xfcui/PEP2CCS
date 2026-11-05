<p align="right"><a href="../README.md">README</a> · <b>中文</b> | <a href="README.en.md">English</a></p>

# 权重

仓库里提交的是 `archives/` 下的 zip，避免单个权重目录过大。`.pt` 本身已经是压缩包，再打包大约还能小一成，并且每个 zip 都低于 GitHub 的单文件限制。第一次加载时，程序按下面的相对路径把需要的文件解压到本目录。这些是 pickle，只加载你信任的文件。预测使用其中的 `model_param`。

| 压缩包 | 内容 |
|---|---|
| archives/pep2ccs.zip | pep2ccs.pt |
| archives/exp2.zip | exp2/pep2ccs.pt、exp2/lstm.pt |
| archives/exp3.zip | exp3/model1.pt … model3.pt |
| archives/exp4.zip | exp4/model.pt |
| archives/exp5-part1.zip | exp5/model1.pt … model3.pt |
| archives/exp5-part2.zip | exp5/model4.pt … model6.pt |
| archives/exp5-part3.zip | exp5/model7.pt |

| 解压后的文件 | 结构 |
|---|---|
| pep2ccs.pt | 主模型，Exp1 和 Exp2 |
| exp2/pep2ccs.pt | Exp2 的 PEP2CCS 权重。词嵌入 50 行，回归输入 259，结构见 [实验说明](../docs/experiments.md) |
| exp2/lstm.pt | Exp2 的 3 层双向 LSTM |
| exp3/model1.pt | m/z 写入 embedding |
| exp3/model2.pt | 长度写入 embedding |
| exp3/model3.pt | 电荷、长度、m/z 写入 embedding |
| exp4/model.pt | 组成向量模型 |
| exp5/model1.pt … model7.pt | 七种 pooling，顺序与 [实验说明](../docs/experiments.md) 相同 |

产生这些文件的 PyTorch 版本没有记录在原始仓库里。本仓库用 PyTorch 2.2.2 加载。
