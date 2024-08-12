[**English**](./README.md) | [**中文**](./README_zh.md)

1. 运行 `bash 1_get_cache.sh`，加载并且tokenize所有的数据，并且存储在本地的 `tokenized_path`。
2. 运行 `bash 2_start_pretrain.sh` 开始预训练, `tokenized_path` 参数需要与 `1_get_cache.sh`中设置的一致，预训练后的模型将存在 `output_dir`位置。
