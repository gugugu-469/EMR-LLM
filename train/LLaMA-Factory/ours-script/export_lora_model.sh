cd ../

python src/export_model.py \
    --model_name_or_path path_of_base_llm \
    --template chatglm3 \
    --adapter_name_or_path lora_path \
    --lora_target query_key_value \
    --finetuning_type lora \
    --export_dir export_path \
    --export_size 3 \
    --export_legacy_format False

