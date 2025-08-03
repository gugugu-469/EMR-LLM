import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch




# src_path=[
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs/chatglm3-6b-r1-countdown_lora_merge_1e-5/checkpoint-200',
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs/chatglm3-6b-r1-countdown_lora_merge_1e-5/checkpoint-400',
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs/chatglm3-6b-r1-countdown_lora_merge_1e-5/checkpoint-608',
# ]
# export_path=[
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs_export/chatglm3-6b-r1-countdown_lora_merge_1e-5_ck200',
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs_export/chatglm3-6b-r1-countdown_lora_merge_1e-5_ck400',
# '/HL_user01/2025-03-27-GRPO/deep-learning-pytorch-huggingface_scp/training/runs_export/chatglm3-6b-r1-countdown_lora_merge_1e-5_ck608',
# ]

src_path=[
'/HL_user01/2025-03-27-GRPO/2025-06-17-qcic模型_数据集/qcic_最终的数据和模型/0622_模型/checkpoint-8000',
]
export_path=[
'/HL_user01/2025-03-27-GRPO/2025-06-17-qcic模型_数据集/qcic_最终的数据和模型/0622_模型/checkpoint-8000_export',
]
model_name = "/HL_user01/trained_models/0229_ck36000_sft_stage4_lora_04-28-16-01-02_ck38000_export_model"



for lora_model_path, save_path in zip(src_path, export_path):
    # 1. 加载原始模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 根据硬件调整
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

    # 3. 合并模型（将LoRA权重合并到原始模型）
    merged_model = lora_model.merge_and_unload()  # 关键步骤！

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"模型已合并并保存到: {save_path}")