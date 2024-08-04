cd ./evaluate_code
seed=2024
gpu=0

dataset_dir=path_of_medbench

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

# chatglm3
model="path_of_chatglm3"
model_dtype="fp16"
template="chatglm3"
split_max_len=8192

# BenTaso
model="path_of_bentaso"
model_dtype="fp16"
template="BenTaso"
split_max_len=8192

# Huatuo-7b
model="path_of_huatuo_7b"
model_dtype="bf16"
template="Huatuo"
split_max_len=4096

# Huatuo-13b
model="path_of_huatuo_13b"
model_dtype="bf16"
template="Huatuo"
split_max_len=4096

# Zhongjing
model="path_of_zhongjing"
model_dtype="fp16"
template="Zhongjing"
split_max_len=2048

# AlpaCare
model="path_of_alpacare"
model_dtype="fp16"
template="AlpaCare"
split_max_len=4096

# pmc_llama
model="path_of_pmc_llama"
model_dtype="fp16"
template="pmc_llama"
split_max_len=2048

# emr-llm
model="path_of_emr_gpt"
model_dtype="fp16"
template="chatglm3"
split_max_len=8192


echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_medbench.py \
    --model_name_or_path ${model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}
