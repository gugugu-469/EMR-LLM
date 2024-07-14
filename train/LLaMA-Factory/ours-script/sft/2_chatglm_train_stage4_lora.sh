export WANDB_DISABLED=True
now_time=$(command date +%m-%d-%H-%M-%S)
echo "now time ${now_time}"
cd ../../
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=20101 src/train_bash.py --deepspeed ./ours-script/ds_config_zero2.json \
    --stage sft \
    --model_name_or_path path_of_base_llm \
    --do_train \
    --do_eval \
    --dataset dataset_1,dataset_2,... \
    --lora_rank 32 \
    --lora_alpha 64 \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir path_to_save_stage4_model \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
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
# nohup bash 2_chatglm_train_stage4_lora.sh > 2_chatglm_train_stage4_lora.out & disown