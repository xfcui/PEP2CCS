<p align="right"><a href="../README.md">README</a> · <a href="README.md">中文</a> | <b>English</b></p>

# Example data

`test_data.zip` is the test table shipped with the release. Extracting it produces `test_data.csv`. Scripts extract the archive when the CSV is absent. The extracted file does not need to be committed.

Column definitions are in the [data schema](../docs/data_schema.en.md).

The training set was not released with the repository. Pass your own CSV to `scripts/train.py`. The split is `train_test_split(test_size=0.01, random_state=1115)`.
