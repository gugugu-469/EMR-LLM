# coding=utf-8
# Evaluates the performance of pre-trained models.
# Usage: python evaluate.py --model_name_or_path path_to_model --checkpoint_dir path_to_ckpt --template vanilla
#                           --task ceval --split validation --lang zh --n_shot 5 --batch_size 4 --save_name result
# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import jsonlines
from misc import *
from types import MethodType
import os
from template import get_template_and_fix_tokenizer
import re
import fire
import json
import torch
import time
from datetime import datetime, timedelta
import numpy as np
import transformers
from collections import Counter, defaultdict
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm, trange
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoModel
if TYPE_CHECKING:
    from datasets import Dataset



# dataclass会自动生成__init__,__repr__等
@dataclass
class EvalTemplate:

    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(
        self,
        example: Dict[str, str],
        choices: List
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example and example[ch] != 'nan']
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        subject_name: str,
        use_history: bool,
        choices: List
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example(target_data,choices=choices)
        history = [self.parse_example(support_set[k],choices=choices) for k in range(len(support_set))]

        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.system.format(subject=subject_name) + temp[0], temp[1]))
        else:
            query = self.system.format(subject=subject_name) + query

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates = {
    "en": EvalTemplate(
        system="Please output the answer choices directly without any analysis or interpretation.\n\n",
        choice="\n{choice}. {content}",
        answer="\nAnswer: ",
        prefix=" "
    ),
    "zh": EvalTemplate(
        system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。不需要做任何分析和解释，直接输出选项字母。\n\n",
        choice="\n{choice}： {content}",
        answer="\n答案：",
        prefix="\n"
    ),
    "zh_multiple": EvalTemplate(
        system="以下是中国关于{subject}考试的不定项选择题，题目有一个或多个答案，请选出其中的正确答案。不需要做任何分析和解释，直接输出选项字母。\n\n",
        choice="\n{choice}： {content}",
        answer="\n答案：",
        prefix="\n"
    ),
}

def get_time(fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间，并增加8小时
    """
    # 获取当前时间
    ts = time.time()
    current_time = datetime.fromtimestamp(ts)

    # 增加8小时
    adjusted_time = current_time + timedelta(hours=8)

    # 格式化时间
    return adjusted_time.strftime(fmt)


def evaluate(
    model_name_or_path: str,
    finetuning_type: Optional[str] = "lora",
    model_dtype: Optional[str] = 'fp16',
    checkpoint_dir: Optional[str] = None,
    template: Optional[str] = "chatglm3",
    task: Optional[str] = "ceval",
    dataset_dir: Optional[str] = "evaluation",
    split: Optional[Literal["validation", "test"]] = "validation",
    lang: Optional[Literal["zh", "en"]] = "zh",
    n_shot: Optional[int] = 5,
    n_avg: Optional[int] = 1,
    batch_size: Optional[int] = 4,
    seed: Optional[int] = 42,
    output_dir: Optional[str] = "../gen_choice_output",
    use_history: Optional[Literal[False,True]] = True,
    use_vllm: Optional[Literal[False,True]] = True,
    print_nums: Optional[int] = 5,
    split_max_len: Optional[int] = 8192,
):
    

    if 'CMB' in dataset_dir or 'CMB' in task:
        choices = ["A", "B", "C", "D","E"]
    else:
        choices = ["A", "B", "C", "D"]
    
    start_time = time.time()
    out_time = get_time('%m-%d-%H-%M-%S')
    print('out_time:{}'.format(out_time))
    # 以模型名称为文件夹，一个模型的所有预测文件放在一起
    model_name = model_name_or_path.split('/')[-1]
    output_dir = os.path.join(output_dir,model_name,'commons_gen_choice_{}'.format(use_history))
    output_dataset_name = '{}|{}|{}|{}|{}'.format(task,split,lang,n_shot,out_time)
    output_dir = os.path.join(output_dir,output_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('use_history:{}'.format(use_history))
    # 各个题目主题以及对应中文
    print('mapping:{}'.format(os.path.join(dataset_dir, task, "mapping.json")))
    with open(os.path.join(dataset_dir, task, "mapping.json"), "r", encoding="utf-8") as f:
        categorys: Dict[str, Dict[str, str]] = json.load(f)

    # 从mapping中读取所有的类别，并加上AVERAGE作为总体的评估
    all_categorys = ['Average']
    for k,v in categorys.items():
        now_category = v['category']
        if now_category not in all_categorys:
            all_categorys.append(now_category)
    print('所有的大类:{}'.format(all_categorys))
    print('seed:{}'.format(seed))
    # seed
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    if model_dtype == 'fp16':
        use_type = 'float16'
    elif model_dtype == 'bf16':
        use_type = 'bfloat16'
    else:
        use_type = model_dtype
        
    # 获得模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,padding_side="right")
    if template == 'chatglm3':
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
        tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
    # self.tokenizer.padding_side = "left"
    model_template = get_template_and_fix_tokenizer(tokenizer, template)
    if template == 'chatglm3':
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|user|>"), tokenizer.convert_tokens_to_ids("<|observation|>")]
    elif template == 'medalpaca':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'mmedlm':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'pmc_llama':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'llama3':
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif template == 'BenTaso':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'Huatuo':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'Zhongjing':
        eos_token_id = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<bot>'),tokenizer.convert_tokens_to_ids('<human>')]
    elif template == 'BianQue':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'AlpaCare':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'Meditron':
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')]

    if use_vllm:
        use_model = False
        print('use_type:{}'.format(use_type))
        model = LLM(model=model_name_or_path,tokenizer_mode='auto', trust_remote_code = True,dtype=use_type, max_model_len = split_max_len, gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.8, max_tokens = split_max_len, stop_token_ids=eos_token_id, seed=seed)
    else:
        if use_type == 'fp16':
            model_dtype = torch.float16
        elif use_type == 'bf16':
            model_dtype = torch.bfloat16
        else:
            model_dtype = 'error'
            print('dtype error!!!')
        if template == 'BianQue':
            model = AutoModel.from_pretrained(model_name_or_path,torch_dtype=model_dtype, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=model_dtype, trust_remote_code=True)
        model.eval()
        model = dispatch_model(model)
    print('use_history:{}'.format(use_history))

    # 针对多选题，使用多选题数据构造模板
    if 'CMB_Exam_processed_choice_gen' in dataset_dir or 'CMB_Exam_processed_choice_gen' in task:
        eval_template = eval_templates['zh_multiple']
    else:
        eval_template = eval_templates[lang]

    # 以学科为单位统一(比如abstract_algebra、anatomy等)
    subject_corrects = defaultdict(list)

    # 大类的方法统一， 大类指:inter/spec(Xiezhi数据集)；STEM、(MMLU数据集)
    category_corrects: Dict[str, np.ndarray] = {
        subj: [] for subj in all_categorys
    }

    pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
    results_category = {}
    # 一个学科一个学科
    print('\n总共跑{}次，取平均值'.format(n_avg))
    print('评测集合:{}'.format(split))
    for subject in pbar:

        print('加载数据集:{}\n学科:{}'.format(os.path.join(dataset_dir, task),subject))
        load_time = time.time()
        dataset = load_dataset(os.path.join(dataset_dir, task), subject)
        load_end_time = time.time()
        print('load takes:{} seconds'.format(load_end_time-load_time))

        if 'xiezhi' in dataset_dir or 'xiezhi' in task:
            data_indices_by_label = defaultdict(list)
            # 遍历数据集
            for i, item in enumerate(dataset['train']):
                # 假设 labels 是以 '||' 分隔的字符串
                for label in item['labels'].split('||'):
                    # 将数据的索引添加到对应的标签下
                    data_indices_by_label[label].append(i)



        labels, answers, all_outputs = [], [], []
        # 多次取平均值
        for epoch in range(n_avg):
            preds = []
            prompts = []
            # 可以预测的下标
            corrects = {}
            # 不可以预测的下标
            errors = []
            


            # 给进度条设定前缀
            pbar.set_postfix_str("{} Trial: {}".format(categorys[subject]["name"], epoch))
            inputs, outputs = [], []
            # 跑test集合还是dev集合
            # position是第几行打印进度条，因为上面还有一个pbar，所以设置为1
            for i in trange(len(dataset[split]), desc="Formatting batches", position=1, leave=False):
                # 选定n_shot例子
                sample = {}
                if 'xiezhi' in dataset_dir or 'xiezhi' in task:
                    # 因为xiezhi的一条数据里面有多个学科，一级学科、二级学科、三级学科等，遍历每个学科，取最少的学科集条数，认为是最细的学科
                    # 也可以 一级 二级 三级 统计完以后分别匹配
                    now_labels = dataset[split][i]['labels']
                    min_num = 99999
                    for label in now_labels.split('||'):
                        if len(data_indices_by_label[label]) < min_num:
                            min_num = len(data_indices_by_label[label])
                            min_label = label
                    subset = dataset['train'].select(data_indices_by_label[min_label])
                else:
                    # 其他数据集，都是一个学科一个学科分割好的
                    if 'train' in dataset.keys():
                        subset = dataset["train"]
                    else:
                        subset = []
                
                # 针对中医综合，特殊处理
                if ('CMB' in dataset_dir or 'CMB' in task) and subject == '中医综合':
                    select_num = 0
                else:
                    select_num = min(n_shot, len(subset))
                
                if select_num == 0:
                    support_set = []
                else:
                    support_set = dataset["train"].shuffle().select(range(select_num))
                
                if 'xiezhi' in dataset_dir or 'xiezhi' in task:
                    # query 提问 ,resp 答案， history是 n_shot(List((query,resp)))
                    query, resp, history = eval_template.format_example(
                        target_data=dataset[split][i],
                        support_set=support_set,
                        subject_name=min_label,
                        use_history=use_history,
                        choices = choices
                    )
                else:
                    # query 提问 ,resp 答案， history是 n_shot(List((query,resp)))
                    query, resp, history = eval_template.format_example(
                        target_data=dataset[split][i],
                        support_set=support_set,
                        subject_name=categorys[subject]["name"],
                        use_history=use_history,
                        choices = choices
                    )
                sample['query'] = query
                sample['resp'] = resp
                sample['history'] = history
                # 构造messages
                messages = []
                # 先放入history
                for his in sample['history']:
                    messages.append({
                        'role':'user', 'content':his[0]
                    })
                    messages.append({
                        'role':'assistant', 'content':his[1]
                    })
                # 放入当前的
                messages.append({
                    'role':'user', 'content': sample['query']
                })
                messages.append({
                    'role':'assistant', 'content': sample['resp']
                })
                now_input_token, _ = model_template.encode_oneturn(tokenizer,messages)

                # if template == 'chatglm3':
                #     now_input_token = tokenizer.build_chat_input(query, history = history)['input_ids'][0].tolist()
                # elif template == 'llama3':
                #     messages = [
                #         {"role": "system", "content": "请用中文回答"},
                #         {"role": "user", "content": query},
                #     ]
                #     now_input_token = tokenizer.apply_chat_template(messages)
                # else:
                #     prompt = prompt[0].format(query)
                #     now_input_token = tokenizer(prompt)["input_ids"]

                tmp_input_len = len(now_input_token)
                # 如果是第一轮，加入答案
                if epoch == 0:
                    labels.append(resp)
                    # 第一轮，打印一下
                    if i < print_nums:
                        now_print_dir = os.path.join(output_dir,'打印',str(i))
                        if not os.path.exists(now_print_dir):
                            os.makedirs(now_print_dir)
                        with open(os.path.join(now_print_dir,'query.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(query))
                        with open(os.path.join(now_print_dir,'resp.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(resp))
                        with open(os.path.join(now_print_dir,'history.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(history))
                        with open(os.path.join(now_print_dir,'encode.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(now_input_token))
                        with open(os.path.join(now_print_dir,'decode.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(tokenizer.decode(now_input_token)))

                if use_vllm:
                    # 如果使用vllm，先全部存起来
                    if tmp_input_len >= split_max_len:
                        errors.append(i)
                        prompts.append(now_input_token)
                        sample['over_length'] = True
                    else:
                        # 只有在长度内的才放进来
                        corrects[i] = len(corrects.keys())
                        prompts.append(now_input_token)
                        sample['over_length'] = False
                    sample['prompt'] = now_input_token
                    sample['prompt_len'] = len(now_input_token)
                    preds.append(sample)
                else:
                    # 否则直接预测就行
                    # 如果还是超过最大长度，跳过
                    if tmp_input_len >= split_max_len:
                        print('index:{} 超过最大长度 长度为:{} 模型最大长度:{}'.format(i,tmp_input_len,split_max_len))
                        response = '##ERROR##超过最大长度'
                        sample['over_length'] = True
                    else:
                        # 使用原版chat时，有时会将答案自动解析
                        response = mine_chat(tokenizer, now_input, history=model_history)
                        sample['over_length'] = False
                    sample['pred'] = response
                    sample['prompt'] = now_input_token
                    sample['prompt_len'] = len(now_input_token)
                    preds.append(sample)
                    outputs.append(re.sub(r'\s','',response))
            # 使用vllm，就统一评估
            if use_vllm:
                with jsonlines.open(os.path.join(output_dir,"processed_prompts_{}_ep{}.json".format(task,epoch)),'w') as f:
                    for s_index,sample in tqdm(enumerate(preds)):
                        f.write(sample)
                # vllm预测
                now_outputs = model.generate(prompt_token_ids = prompts, sampling_params = sampling_params)
                # 输出
                correct_indexes = corrects.keys()
                # 赋值pred即可
                for s_index,sample in tqdm(enumerate(preds)):
                    if s_index in correct_indexes:
                        output_index = corrects[s_index]
                        response = now_outputs[output_index].outputs[0].text
                    else:
                        assert s_index in errors
                        response = '##ERROR##超过最大长度'
                    sample['pred'] = response
                    outputs.append(re.sub(r'\s','',response))


            all_res_output_dir = os.path.join(output_dir,'打印')
            if not os.path.exists(all_res_output_dir):
                os.makedirs(all_res_output_dir)
            all_res_output_file = os.path.join(all_res_output_dir,'{}_{}.jsonl'.format(subject,epoch))
            with jsonlines.open(all_res_output_file,'w') as f:
                for sample in preds:
                    f.write(sample)


            all_outputs.append(outputs)

        for i in range(len(all_outputs[0])):
            count = Counter([all_outputs[epoch][i] for epoch in range(n_avg)])
            # 多次取最常出现的
            # 参数 most_common(arg) arg表示拿几个，取最大的1个，所以设为1，返回的为一个list[tuple], tuple表示(值，出现次数)
            answers.append(count.most_common(1)[0][0])

        corrects = (np.array(answers) == np.array(labels)).tolist()
        # 加入average中
        category_corrects["Average"] = category_corrects["Average"] + corrects
        
        # 根据数据集单独判断下
        if 'xiezhi' in dataset_dir or 'xiezhi' in task:
            # 每条数据的label可能不同，所以一一遍历判断下
            for i in range(len(dataset[split])):
                # label在下面的labels字段中，一条数据可能有多个学科，通过|| 分割，比如“历史学||文学||新闻学||新闻传播学”
                now_subjects = dataset[split][i]['labels']
                for tmp_subject in now_subjects.split('||'):
                    subject_corrects[tmp_subject] = subject_corrects[tmp_subject] + [corrects[i]]
        else:
            # MMLU中，已经按照subject对文件分割了
            tmp_subject = subject
            subject_corrects[tmp_subject] = subject_corrects[tmp_subject] + corrects

        # 大类的实现方法一样
        category_name = categorys[subject]["category"]
        category_corrects[category_name] = category_corrects[category_name] + corrects
        results_category[category_name] = {str(i): answers[i] for i in range(len(answers))}
        # print('category_corrects:{}'.format(category_corrects))
        

    # Category
    score_info = "\n".join([
        "{:>15}: 个数:{} 得分:{:.2f}".format(category_name, len(category_correct),100 * np.mean(category_correct))
        for category_name, category_correct in category_corrects.items() if len(category_correct)
    ])
    avg_score = 100 * np.mean(category_corrects['Average'])
    print('Category score')
    print(score_info)

    with open(os.path.join(output_dir,"category_score_info.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write('{}\n'.format(avg_score))
        f.write(score_info)

    # Subject
    score_info = "\n".join([
        "{:>15}: 个数:{} {:.2f}".format(subject_name, len(subject_correct), 100 * np.mean(subject_correct))
        for subject_name, subject_correct in subject_corrects.items() if len(subject_correct)
    ])
    print('Subject score')
    print(score_info)

    with open(os.path.join(output_dir,"subject_score_info.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write(score_info)


    with open(os.path.join(output_dir,"preds.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(results_category, f, indent=2,ensure_ascii=False)
    end_time = time.time()
    print('cost time:{}'.format(end_time-start_time))
    with open(os.path.join(output_dir,'args.txt'),'w',encoding='utf-8') as f:
        f.write('out_time:{}\n'.format(out_time))
        f.write('model_name_or_path:{}\n'.format(model_name_or_path))
        f.write('template:{}\n'.format(template))
        f.write('task:{}\n'.format(task))
        f.write('dataset_dir:{}\n'.format(dataset_dir))
        f.write('split:{}\n'.format(split))
        f.write('lang:{}\n'.format(lang))
        f.write('n_shot:{}\n'.format(n_shot))
        f.write('n_avg:{}\n'.format(n_avg))
        f.write('seed:{}\n'.format(seed))
        f.write('use_history:{}\n'.format(use_history))
        f.write('cost time:{}\n'.format(end_time-start_time))


if __name__ == "__main__":
    fire.Fire(evaluate)
