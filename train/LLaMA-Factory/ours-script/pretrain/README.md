[**English**](./README.md) | [**中文**](./README_zh.md)

1. run `bash 1_get_cache.sh` to load and tokenize the datasets, then it will save the tokenized datast to `tokenized_path`.
3. run `bash 2_start_pretrain.sh` to start pretraining, make sure the `tokenized_path` parameter is the same as the one in `1_get_cache.sh`, the pretrained model will be saved at `output_dir`.
