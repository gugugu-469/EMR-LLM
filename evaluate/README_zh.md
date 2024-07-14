[**English**](./README.md) | [**中文**](./README_zh.md)
# 评估预训练效果
## 添加提示并评估
数据集：C-Eval,CMMLU,MMLU,CMB_Exam,MedQA-MCMLE,Med-C-Eval,Med-CMMLU,Med-MMLU

运行 `run_eval_pt_choice_add_prompt.sh` 对这些数据集进行评估。

在评估过程中，我们将添加特殊提示，并将样本格式化为标准字符串文本。

我们会根据输入提示计算每个答案选项的对数概率，并选择概率最高的选项作为模型的答案。
### 在其他数据集上进行评估
### GSM8K
运行 `run_eval_pt_gsm8k.sh` 在 GSM8K 上进行评估。

由于该数据集中没有选项，我们以预训练格式标记问题，然后使用模型生成答案。生成答案后，我们从预测中提取答案，并将提取的答案与正确数据进行比较。
#### GAOKAO
运行 `run_eval_pt_choice_gaokao.sh` 在 GAOKAO 上进行评估。
### AGIEval
运行 `run_eval_pt_choice_agieval.sh` 在 GAOKAO 上进行评估。

我们以选择题的格式处理数据集，并记录不同类型问题的选项数量。
#### BBH
运行 `run_eval_pt_choice_bbh.sh` 在 GAOKAO 上进行评估。

我们以选择题的格式处理数据集，并记录了不同类型问题的选项数。
# 评估指令微调效果
## 评估基准指标
数据集：C-Eval,CMMLU,MMLU,CMB_Exam,MedQA_USMLE,Med-C-Eval,Med-CMMLU,Med-MMLU

我们将问题标记为对话格式，然后使用模型生成答案。生成答案后，我们从预测中提取选项，并将提取的选项与正确选项进行比较。
## 在 EMR 任务上进行评估
数据集：指令 EMR

我们将问题标记为对话格式，然后使用模型生成答案。我们计算指标的方法取决于数据类型。

## 环境

在训练环境上，额外安装了`vllm==0.4.1`用于加速