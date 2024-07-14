cd ./evaluate_code
seed=2024
gpu=0

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

dataset_dir=path_of_pt_eval_datas
datasets=(BBH)
langs=(en)
splits=(test)
n_shots=(0)
length=${#datasets[@]}

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

# emr-gpt
model="path_of_emr_gpt"
model_dtype="fp16"
template="chatglm3"
split_max_len=8192

# 运行
for (( i=0; i<$length; i++ )); do
  dataset=${datasets[$i]}
  lang=${langs[$i]}
  split=${splits[$i]}
  for n_shot in "${n_shots[@]}"
  do
    CUDA_VISIBLE_DEVICES=${gpu} python -u eval_common_bbh.py \
      --model_name_or_path ${model} \
      --lang ${lang} \
      --dataset_dir ${dataset_dir} \
      --template ${template} \
      --model_dtype ${model_dtype} \
      --task ${dataset} \
      --split ${split} \
      --lang ${lang} \
      --n_shot ${n_shot} \
      --n_avg 1 \
      --seed ${seed} \
      --batch_size 2
  done
done
