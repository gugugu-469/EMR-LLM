# Evaluate Pretrain Performance
## Evaluate with prompt added
Dataset:C-Eval,CMMLU,MMLU,CMB_Exam,MedQA-MCMLE,Med-C-Eval,Med-CMMLU,Med-MMLU

run `run_eval_pt_choice_add_prompt.sh` to evaluate on these datasets.

In the evaluation process, we will add special prompts and format the sample into standard string text.

We compute the log-likelihood for each answer option following the input prompt and select the highest probability option as the model’s response.
## Evaluate on other datasets
### GSM8K
run `run_eval_pt_gsm8k.sh` to evaluate on GSM8K.

Since there are no options in this dataset, we tokenize the question in pretrain format and and use the model to generate answer. After generation, we extract the answers from the predictions and compare the extracted answer with gold data.
### GAOKAO
run `run_eval_pt_choice_gaokao.sh` to evaluate on GAOKAO.
### AGIEval
run `run_eval_pt_choice_agieval.sh` to evaluate on GAOKAO.

We process the dataset in the format of multiple-choice questions and recorded the number of options that different types of questions had.
### BBH
run `run_eval_pt_choice_bbh.sh` to evaluate on GAOKAO.

We process the dataset in the format of multiple-choice questions and recorded the number of options that different types of questions had.
# Evaluate SFT Performance
## Evaluate on BenchMarks
Dataset:C-Eval,CMMLU,MMLU,CMB_Exam,MedQA_USMLE,Med-C-Eval,Med-CMMLU,Med-MMLU

we tokenize the question in chat format and use the model to generate answer. After generation, we extract the option from the predictions and compare the extracted option with the gold one.
## Evaluate on EMR tasks
Dataset:InstructionEMR

we tokenize the question in chat format and use the model to generate answer. The method that we calculate the metric depends on the type of data.

# Environment

Based on the training environment, `vllm==0.4.1` was additionally installed for prediction acceleration.