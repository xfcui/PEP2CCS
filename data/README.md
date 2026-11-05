<p align="right"><a href="../README.zh.md">README</a> · <b>中文</b> | <a href="README.en.md">English</a></p>

# 示例数据

`test_data.zip` 是发布时附带的测试表，解压后为 `test_data.csv`。脚本在文件不存在时自动解压，解压结果不需要提交。

列说明见 [数据约定](../docs/data_schema.md)。

训练集没有随仓库发布。训练时把自己的 CSV 传给 `scripts/train.py`。划分方式是 `train_test_split(test_size=0.01, random_state=1115)`。
