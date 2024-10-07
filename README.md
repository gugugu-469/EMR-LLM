[**English**](./README.md) | [**中文**](./README_zh.md)
# EMR-LLM
This repository is the official implementation of **EMR-LLM**, which is model proposed in a paper: **A Large Language Model for Electronic Medical Records**.

# Brief introduction
we propose **EMR-LLM**, an LLM designed for electronic medical records, capable of fully comprehending the complex structure of EMRs and critical clinical numerical data.

## 🤖 Model
**Firstly**, we collect a substantial amount of open-domain corpora, basic medical corpora, and clinical guidelines as pre-training data to enhance the medical domain knowledge of the LLMs.

**Second**, we gather 75,000 EMRs from Ruijin Hospital which is a top-tier hospital in China. Based on these EMRs, we design three categories of instruction-tuning tasks: structure understanding, numerical understanding, and downstream applications.

**Finally**, we propose an ability-boosting instruction tuning method to fine-tune the pre-trained LLMs with this dataset. This method mimics the human learning process by allowing the LLMs to learn simple tasks first and then gradually progress to more complex tasks.

The framework of EMR-LLM is shown in the following figure:
<div align="center">
  <img src="assets/framework.jpg" alt="Framework" width="100%">
</div>

# 🔬 Requirements

To run our code, please install dependency packages.
```
accelerate	    0.27.2
deepspeed	    0.14.2
fire	        0.5.0
flash-attn	    2.5.8
ninja	        1.11.1.1
sentencepiece	0.1.99
torch	        2.2.1
vllm	        0.4.1
peft	        0.10.0
trl	            0.8.1
datasets    	2.17.1	
transformers	4.40.0	
scipy	        1.12.0
tiktoken    	0.6.0	
protobuf	    3.20.3	
pydantic    	2.6.1	
matplotlib	    3.8.3	
sse-starlette	2.0.0	
packaging	    23.2	
pyyaml      	6.0.1
pandas	        1.5.3
numpy	        1.23.4
```

# Code Structure

`./train`:The dictory of training code.

`./evaluate`:The directory of evaluation code.

`dataset`:The directory of sample dataset.

# 🚀 Quick start

If you want to be consistent with our build process, go to the `/train/LLaMA-Factory/ours-script` directory and follow the instructions in the directory.

## Pre-training
```sh
# go to the directory
cd /train/LLaMA-Factory/ours-script/pretrain

# get the dataset cache
bash 1_get_cache.sh

# start pre-training
bash 2_start_pretrain.sh
```

## SFT
```sh
# go to the directory
cd /train/LLaMA-Factory/ours-script/sft

# get the dataset cache of stage1 to stage4
bash 1_chatglm_cache_stage1.sh
bash 1_chatglm_cache_stage2.sh
bash 1_chatglm_cache_stage3.sh
bash 1_chatglm_cache_stage4.sh

# start sft stage by stage
bash 2_chatglm_train_stage1_lora.sh
# Modify Configuration
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage2_lora.sh
# Modify Configuration
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage3_lora.sh
# Modify Configuration
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage4_lora.sh
# Modify Configuration
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
```

# Acknowledgements

Special thanks to [hiyouga](https://github.com/hiyouga/LLaMA-Factory) for providing the LLaMA fine-tuning framework.

This project is based on [Chatglm3-6b](https://github.com/THUDM/ChatGLM3).

Thanks to the publishers for all opens-ource datasets. 

# About
Should you have any questions, please feel free to contact us at `y80220109@mail.ecust.edu.cn`
