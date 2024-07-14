import os
from misc import *
import torch
import fire
from template import get_template_and_fix_tokenizer
from transformers import GenerationConfig, TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoModel
import transformers
from types import MethodType
import json
import jsonlines
from collections import defaultdict
import random
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from vllm import LLM, SamplingParams
from template import get_template_and_fix_tokenizer

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
    dataset_dir: Optional[str] = "../数据",
    model_dtype: Optional[str] = 'fp16',
    template: Optional[str] = 'chatglm3',
    n_shot: Optional[int] = 2,
    n_avg: Optional[int] = 1,
    seed: Optional[int] = 42,
    output_dir: Optional[str] = "../output",
    predict_nums: Optional[int] = -1,
    use_vllm: Optional[Literal[False,True]] = False,
    split_max_len: Optional[int] = 8192,
):
    output_dir = '../output_emr'
    out_time = get_time('%m-%d-%H-%M-%S')
    print('out_time:{}'.format(out_time))
    # 以模型名称为文件夹，一个模型的所有预测文件放在一起
    model_name = model_name_or_path.split('/')[-1]
    output_dir = os.path.join(output_dir,model_name,'medical')
    output_dataset_name = '{}|{}|{}|{}'.format('emr',predict_nums,n_shot,out_time)
    output_dir = os.path.join(output_dir,output_dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 加载模型
    transformers.set_seed(seed)
    random.seed(seed)
    if model_dtype == 'fp16':
        use_type = 'float16'
    elif model_dtype == 'bf16':
        use_type = 'bfloat16'
    else:
        use_type = model_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print('template:{}'.format(template))
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
        eos_token_id = [tokenizer.eos_token_id, 21529, 102468]
    elif template == 'Huatuo':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'Zhongjing':
        eos_token_id = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<bot>'),tokenizer.convert_tokens_to_ids('<human>')]
    elif template == 'BianQue':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'AlpaCare':
        eos_token_id = [tokenizer.eos_token_id]
    elif template == 'qwen':
        eos_token_id = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|endoftext|>'),tokenizer.convert_tokens_to_ids('<|im_start|>'),tokenizer.convert_tokens_to_ids('<|im_end|>')]
    elif template == 'Meditron':
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')]

    if use_vllm:
        model = LLM(model=model_name_or_path,tokenizer_mode='auto', trust_remote_code = True,dtype=use_type, max_model_len = split_max_len, gpu_memory_utilization=0.9)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.7, repetition_penalty=1.1, max_tokens = split_max_len, stop_token_ids=eos_token_id, seed=seed)
    else:
        if use_type == 'fp16':
            model_dtype = torch.float16
        elif use_type == 'bf16':
            model_dtype = torch.bfloat16
        else:
            model_dtype = 'error'
            raise ValueError('dtype error!!!')
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=model_dtype, trust_remote_code=True)
        model.eval()
        model = dispatch_model(model)

    # 加载数据
    # 遍历文件夹获得文件
    if os.path.isdir(dataset_dir):
        files = []
        # 如果是文件夹，遍历获得所有可能的测试文件
        keshis = os.listdir(dataset_dir)
        for keshi in keshis:
            keshi_dir = os.path.join(dataset_dir,keshi)
            for file in os.listdir((keshi_dir)):
                if not (file.endswith('jsonl') and 'test' in file):
                    continue
                file_path = os.path.join(keshi_dir,file)
                files.append(file_path)
    else:
        # 如果是文件，仅获取此文件即可
        files = [dataset_dir]
    start_time = time.time()
    for file in files:
        file_start = time.time()
        # 拿到test集，并且是jsonl文件
        print('now file:{}'.format(file))
        train_file = file

        keshi = os.path.basename(os.path.dirname(train_file))

        # 获取文件名和后缀
        file_name_with_extension = os.path.basename(train_file)
        task, file_extension = os.path.splitext(file_name_with_extension)


        print('task:{}'.format(task))
        tmp_dir = os.path.join(output_dir,keshi,task)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        print('write to:{}'.format(tmp_dir))
        test_samples = []
        with jsonlines.open(os.path.join(dataset_dir,file),'r') as f:
            for data in f:
                test_samples.append(data)

        if predict_nums == -1:
            predict_datas = test_samples
        elif predict_nums <= 0:
            raise ValueError("predict nums设定错误，应为-1或大于0")
        else:
            predict_datas = test_samples[:predict_nums]
        
        preds = []
        prompts = []
        # 可以预测的下标
        corrects = {}
        # 不可以预测的下标
        errors = []
        for s_index,sample in tqdm(enumerate(predict_datas)):


            messages = []
            # 放入当前的
            messages.append({
                'role':'user', 'content': sample['instruction']
            })
            messages.append({
                'role':'assistant', 'content': sample['output']
            })
            now_input_token, _ = model_template.encode_oneturn(tokenizer,messages)
            tmp_input_len = len(now_input_token)

            if use_vllm:
                # 如果使用vllm，先全部存起来
                if tmp_input_len >= split_max_len:
                    errors.append(s_index)
                    # prompts.append(now_input_token)
                    sample['over_length'] = True
                else:
                    # 只有在长度内的才放进来
                    corrects[s_index] = len(corrects.keys())
                    prompts.append(now_input_token)
                    sample['over_length'] = False
                sample['prompt'] = now_input_token
                sample['prompt_len'] = len(now_input_token)
            else:
                # 否则直接预测就行
                # 如果还是超过最大长度，跳过
                if tmp_input_len >= split_max_len:
                    print('index:{} 超过最大长度 长度为:{} 模型最大长度:{}'.format(s_index,tmp_input_len,split_max_len))
                    response = '##ERROR##超过最大长度'
                    sample['over_length'] = True
                else:
                    # 使用原版chat时，有时会将答案自动解析
                    response = mine_chat(model,tokenizer, now_input, history=model_history)
                    # response, history = model.chat(tokenizer, now_input, history=model_history)
                    sample['over_length'] = False
                sample['pred'] = response
                sample['prompt'] = now_input_token
                sample['prompt_len'] = len(now_input_token)
                preds.append(sample)
        # 如果使用vllm，统一预测
        if use_vllm:
            with jsonlines.open(os.path.join(tmp_dir,"processed_prompts.json"),'w') as f:
                for s_index,sample in tqdm(enumerate(predict_datas)):
                    f.write(sample)
            # vllm预测
            outputs = model.generate(prompt_token_ids = prompts, sampling_params = sampling_params)
            # 输出
            correct_indexes = corrects.keys()
            print('得到输出数据')
            # 赋值pred即可
            for s_index,sample in tqdm(enumerate(predict_datas)):
                if s_index in correct_indexes:
                    output_index = corrects[s_index]
                    response = outputs[output_index].outputs[0].text
                else:
                    assert s_index in errors
                    response = '##ERROR##超过最大长度'
                sample['pred'] = response
                preds.append(sample)
        file_end = time.time()
        with jsonlines.open(os.path.join(tmp_dir,"preds.json"),'w') as f:
            for data in preds:
                f.write(data)
            cost_time_dict = {'cost_time':'{}'.format(file_end - file_start)}
            f.write(cost_time_dict)
    end_time = time.time()
    with open(os.path.join(tmp_dir,'args.txt'),'w',encoding='utf-8') as f:
        f.write('out_time:{}\n'.format(out_time))
        f.write('model_name_or_path:{}\n'.format(model_name_or_path))
        f.write('task:{}\n'.format(task))
        f.write('dataset_dir:{}\n'.format(dataset_dir))
        f.write('n_shot:{}\n'.format(n_shot))
        f.write('n_avg:{}\n'.format(n_avg))
        f.write('seed:{}\n'.format(seed))
        f.write('cost time:{}\n'.format(end_time-start_time))




if __name__ == "__main__":
    fire.Fire(evaluate)
