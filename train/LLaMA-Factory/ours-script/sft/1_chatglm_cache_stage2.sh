export WANDB_DISABLED=True
now_time=$(command date +%m-%d-%H-%M-%S)
echo "now time ${now_time}"
cd ../../
deepspeed --include localhost:1 --master_port=20001 src/train_bash.py --deepspeed ./ours-script/ds_config_zero2.json \
    --stage sft \
    --model_name_or_path path_of_base_llm \
    --do_train \
    --do_eval \
    --dataset dataset_1,dataset_2,... \
    --template chatglm3 \
    --finetuning_type full \
    --output_dir path_of_pretrained_model \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --val_size 0.1 \
    --plot_loss \
    --preprocessing_num_workers  48 \
    --bf16 \
    --cutoff_len 8000 \
    --ddp_timeout 180000 \
    --save_total_limit 10 \
    --tokenized_path path_to_save_stage2

# the command to run
# nohup bash 1_chatglm_cache_stage2.sh > record_0229_stage_2_get_cache.out & disown