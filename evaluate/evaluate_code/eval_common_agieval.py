# coding=utf-8
# Evaluates the performance of pre-trained models.
# Usage: python evaluate.py --model_name_or_path path_to_model --checkpoint_dir path_to_ckpt --template vanilla
#                           --task ceval --split validation --lang zh --n_shot 5 --batch_size 4 --save_name result
# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import jsonlines
from misc import *
import os
from template import get_template_and_fix_tokenizer
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoModel
import fire
from types import MethodType
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

if TYPE_CHECKING:
    from datasets import Dataset


five_ops = ['lsat-ar', 'lsat-lr', 'lsat-rc', 'aqua-rat'],
four_ops = ['logiqa-en',
  'sat-math',
  'sat-en',
  'sat-en-without-passage',
  'gaokao-english',
  'logiqa-zh',
  'gaokao-chinese',
  'gaokao-geography',
  'gaokao-history',
  'gaokao-biology',
  'gaokao-chemistry',
  'gaokao-physics',
  'gaokao-mathqa']
english_qa_datasets = ["lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "sat-math", "sat-en", "aqua-rat",
                       "sat-en-without-passage", "gaokao-english"]
chinese_qa_datasets = ["logiqa-zh", "gaokao-chinese", "gaokao-geography", "gaokao-history",
                       "gaokao-biology", "gaokao-chemistry", "gaokao-physics", "gaokao-mathqa"]

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
        choices: List,
        lang: str
    ) -> Tuple[str, str]:
        if len(choices) == 4:
            final_op = 'D'
        else:
            final_op = 'E'
        


        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example and example[ch] != 'nan']
        if 'answer' in example.keys():
            resp = example['answer']
        else:
            resp = ''
        # if lang=='en':
        #     return "Q: "+"".join([example["question"]] +["\nAnswer Choices: "] + candidates + [self.answer.format(final_op = final_op)]), resp
        # else:
        #     return "问题："+"".join([example["question"]] + ["\n选项："]+candidates + [self.answer.format(final_op = final_op)]), resp
        return "".join([example["question"]] +candidates + [self.answer.format(final_op = final_op)]), resp

    def format_example(
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        lang: str,
        subject_name: str,
        choices: List
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example(target_data,choices=choices, lang=lang)
        history = [self.parse_example(support_set[k],choices=choices, lang=lang) for k in range(len(support_set))]

        query = self.system.format(subject=subject_name) + '\n\n'.join(["".join(item) for item in history] + [query])

        return query, resp


eval_templates = {
    "en": EvalTemplate(
        system="The following is a multiple choice question.\n\n",
        choice="\n{choice}. {content}",
        answer="\nAnswer: ",
        prefix=" "
    ),
    "zh": EvalTemplate(
        system="以下是一道单项选择题，请选出其中的正确答案。\n\n",
        choice="\n{choice}： {content}",
        answer="\n答案： ",
        prefix="\n"
    )
}


@torch.inference_mode()
def batch_inference(
    tokenizer: PreTrainedTokenizer,
    model: AutoModelForCausalLM,
    batch_input: Dict[str, torch.Tensor],
    prefix_char: str,
    choices: List,
    batch_size: int
) -> List[str]:
    logits = model(**batch_input).logits
    # 拿到每一个的实际长度
    if batch_size == 1:
        lengths = [len(batch_input['input_ids'][0])]
    else:
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
    # 根据长度拿到对应的最后一个字的logits
    nextword_logits = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
    probs = torch.nn.functional.softmax(
        # 每个选项加起来取softmax
        torch.stack(
            [
                nextword_logits[:, tokenizer.encode(prefix_char + choice, add_special_tokens=False)[-1]]
                for choice in choices
            ],
            dim=-1
        ),
        dim=-1
    ).detach()
    # 取概率最大的，并且输出A-D，因为A-D是连着的
    return [chr(ord("A") + offset.item()) for offset in torch.argmax(probs, dim=-1)]

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
    output_dir: Optional[str] = "../common_output",
    print_nums: Optional[int] = 5,
):
    # seed
    transformers.set_seed(seed)
    # 获得模型和tokenizer
    if model_dtype == 'fp16':
        model_dtype = torch.float16
    elif model_dtype == 'bf16':
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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,padding_side="right")
    if template == 'chatglm3' or template == 'BianQue':
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
        tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
    model_template = get_template_and_fix_tokenizer(tokenizer, template)

    tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
    eval_template = eval_templates[lang]

    model_name = model_name_or_path.split('/')[-1]
    base_output_dir = os.path.join(output_dir,model_name,'commons')
    
    # 该部分，不使用chat_template 的模式，仅仅tokenizer问题，通过prob得到答案
    start_time = time.time()
    out_time = get_time('%m-%d-%H-%M-%S')
    print('out_time:{}'.format(out_time))
    # 以模型名称为文件夹，一个模型的所有预测文件放在一起
    output_dataset_name = '{}|{}|{}|{}|{}'.format(task,split,lang,n_shot,out_time)
    output_dir = os.path.join(base_output_dir,output_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
        if subject in five_ops:
            choices = ["A", "B", "C", "D","E"]
        else:
            choices = ["A", "B", "C", "D"]

        if subject in english_qa_datasets:
            subject_lang = 'en'
        else:
            subject_lang = 'zh'

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
            # 给进度条设定前缀
            pbar.set_postfix_str("{} Trial: {}".format(categorys[subject]["name"], epoch))
            inputs, outputs = [], []
            golds = []
            # 跑test集合还是dev集合
            # position是第几行打印进度条，因为上面还有一个pbar，所以设置为1
            for i in trange(len(dataset[split]), desc="Formatting batches", position=1, leave=False):
                # 选定n_shot例子

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
                    query, resp = eval_template.format_example(
                        target_data=dataset[split][i],
                        lang = subject_lang,
                        support_set=support_set,
                        subject_name=min_label,
                        choices = choices
                    )
                else:
                    # query 提问 ,resp 答案， history是 n_shot(List((query,resp)))
                    query, resp = eval_template.format_example(
                        target_data=dataset[split][i],
                        lang = subject_lang,
                        support_set=support_set,
                        subject_name=categorys[subject]["name"],
                        choices = choices
                    )

                input_ids = tokenizer(query)['input_ids']
                
                # print(tokenizer.decode(input_ids))
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                golds.append(dataset[split][i]['answer'])
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
                        with open(os.path.join(now_print_dir,'encode.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(input_ids))
                        with open(os.path.join(now_print_dir,'decode.txt'),'w',encoding='utf-8') as f:
                            f.write('{}'.format(tokenizer.decode(input_ids)))

            for i in trange(0, len(inputs), batch_size, desc="Predicting batches", position=1, leave=False):
                # pad到统一长度，必须在右边补齐，否则会影响logits
                if batch_size == 1:
                    batch_input = {
                        'input_ids': torch.tensor([inputs[i]['input_ids']]).to(model.device)
                    }
                else:
                    batch_input = tokenizer.pad(
                        inputs[i : i + batch_size], return_attention_mask=True, return_tensors="pt"
                    ).to(model.device)
                # batch推理
                preds = batch_inference(tokenizer,model, batch_input, eval_template.prefix, choices, batch_size)
                outputs += preds
            all_outputs.append(outputs)

        for i in range(len(all_outputs[0])):
            count = Counter([all_outputs[epoch][i] for epoch in range(n_avg)])
            # 多次取最常出现的
            # 参数 most_common(arg) arg表示拿几个，取最大的1个，所以设为1，返回的为一个list[tuple], tuple表示(值，出现次数)
            answers.append(count.most_common(1)[0][0])
        all_res = []

        for i in range(len(inputs)):
            input_ids = inputs[i]['input_ids']
            all_res.append({
                'input':tokenizer.decode(input_ids),
                'answer':answers[i],
                'gold':golds[i]
            })
        all_res_output_dir = os.path.join(output_dir,'打印')
        if not os.path.exists(all_res_output_dir):
            os.makedirs(all_res_output_dir)
        all_res_output_file = os.path.join(all_res_output_dir,'{}.jsonl'.format(subject))
        with jsonlines.open(all_res_output_file,'w') as f:
            for res in all_res:
                f.write(res)
            
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
        f.write('batch_size:{}\n'.format(batch_size))
        f.write('seed:{}\n'.format(seed))
        f.write('cost time:{}\n'.format(end_time-start_time))


if __name__ == "__main__":
    fire.Fire(evaluate)
