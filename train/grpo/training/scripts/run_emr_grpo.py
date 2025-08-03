import logging
import os
from dataclasses import dataclass
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datetime import datetime
import logging
import random
import re 
import jsonlines
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from peft import LoraConfig, get_peft_model
import json
import os
import jsonlines
import re
import time
import bert_score
import random
model_path = "/HL_user01/Models/RoBERTa_zh_Large_PyTorch"
mine_bert = bert_score.BERTScorer(model_type=model_path, num_layers=12, device='cuda:2')
def get_bert_scores(pred,gold):
    P, R, F1 = mine_bert.score(pred, gold)
    P = torch.mean(P)
    R = torch.mean(R)
    F1 = torch.mean(F1)
    return P,R,F1
print('devices:{}'.format(torch.cuda.device_count()))
########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


punctuation_map = {
    "，": ",", 
    "。": ".", 
    "！": "!", 
    "？": "?",
    "：": ":", 
    "；": ";", 
    "“": "\"", 
    "”": "\"",
    "‘": "'", 
    "’": "'", 
    "（": "(", 
    "）": ")",
    "【": "[", 
    "】": "]", 
    "《": "<", 
    "》": ">",
    ' ':''
}

def translate_punctuation(text):
    translator = str.maketrans(punctuation_map)
    return text.translate(translator)

def extract_key_value_pairs(data, parent_key='', separator='.'):
    """
    递归提取嵌套字典或列表中的所有键值对
    
    参数:
        data: 要处理的数据(字典或列表)
        parent_key: 父键名(用于构建嵌套键名)
        separator: 嵌套键名之间的分隔符
        
    返回:
        生成器，生成(key, value)元组
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                yield from extract_key_value_pairs(value, new_key, separator)
            else:
                yield (new_key, value)
    elif isinstance(data, list):
        for value in data:  # 不再使用索引
            if isinstance(value, (dict, list)):
                yield from extract_key_value_pairs(value, parent_key, separator)  # 保持父键不变
            else:
                if parent_key:
                    yield (parent_key, value)
                else:
                    # 如果父键为空且值是简单类型，可能需要特殊处理
                    yield (str(value), value)  # 或者可以根据需求调整
########################
# Helper functions
########################
#################################################################################
#通用函数
def flatten_json_extract(data, parent_key=""):
  """
  递归遍历 JSON 数据，分别提取 key-value 对和所有的 key（嵌套的key以 ___ 分隔）
  
  返回：
    pairs: key-value 对列表，格式为 "key||value"
    keys: 所有 key 列表，包含嵌套键链
  """
  pairs = []
  keys = []
  
  if isinstance(data, dict):
      for key, value in data.items():
          # 构建当前键链
          new_key = f"{parent_key}___{key}" if parent_key else key
          # 添加当前键到 keys 列表
          keys.append(new_key)
          
          # 如果 value 为字典，则递归处理
          if isinstance(value, dict):
              sub_pairs, sub_keys = flatten_json_extract(value, new_key)
              pairs.extend(sub_pairs)
              keys.extend(sub_keys)
          # 如果 value 为列表，则对列表内的每个元素处理
          elif isinstance(value, list):
              # 先把列表对应的 key 添加到 keys 列表（已经添加了）
              for element in value:
                  if isinstance(element, dict):
                      sub_pairs, sub_keys = flatten_json_extract(element, new_key)
                      pairs.extend(sub_pairs)
                      keys.extend(sub_keys)
                  else:
                      # 如果列表中的元素不是字典，则视作普通键值对
                      element = re.sub(r'\s','',str(element))
                      pairs.append(f"{new_key}||{element}")
          else:
              # 普通键值对
              value = re.sub(r'\s','',str(value))
              pairs.append(f"{new_key}||{value}")
  elif isinstance(data, list):
      for element in data:
          sub_pairs, sub_keys = flatten_json_extract(element, parent_key)
          pairs.extend(sub_pairs)
          keys.extend(sub_keys)
  else:
      # 其他类型直接输出
      data = re.sub(r'\s','',str(data))
      pairs.append(f"{parent_key}||{data}")
  keys = set(keys)
  pairs = set(pairs)
  return pairs, keys
from rouge_chinese import Rouge
import jieba
rouge = Rouge()
def get_rouge_scores(pred,gold):
    if pred.strip() == '' or gold.strip() == '':
        return 0
    hypothesis = ' '.join(jieba.cut(pred)) 
    reference = ' '.join(jieba.cut(gold))
    scores = rouge.get_scores(hypothesis, reference)
    return scores[-1]['rouge-l']['f']
re_comp = re.compile(r'【.*?】([^【]*)')
re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
re_cankaozhi = re.compile(r'\(.*?\)')

def is_val(text):
    '''
    函数判断是不是指标的数值
    '''
    # 不包含中文，肯定是val
    nums = ['0','1','2','3','4','5','6','7','8','9']
    # if not contains_chinese(text[0]):
    #     return True
    if text[0] in nums:
        return True
    if '阴性' in text or '阳性' in text:
        return True
    return False

def extract_tuples(raw_text):
    raw_text = re_cankaozhi.sub('',raw_text)
    date_clear = re_date.sub('',raw_text)
    texts = re_comp.findall(date_clear)
    if len(texts) == 0:
        texts = [raw_text]
    texts = [text.replace('：','').replace('。','').replace('，',' ').replace('↑','').replace('↓','').replace('；','').replace('','').strip() for text in texts]
    res = []
    for text in texts:
        if ' ' in text:
            text_splits = text.split()
            index = 0
            while index< len(text_splits):
                # 最后一个
                if index + 1 == len(text_splits):
                    merge_text = text_splits[index].replace(' ','')
                    if merge_text != '':
                        res.append(merge_text)
                    index += 1
                    break
                now_text = text_splits[index].strip()
                next_text = text_splits[index+1].strip()
                if is_val(next_text):
                    res.append(str(now_text)+str(next_text))
                    index += 2
                else:
                    if '常规' in now_text or '生化' in now_text:
                        pass
                    else:
                        if now_text != '':
                            res.append(now_text)
                    index += 1

        else:
            if text != '':
                res.append(text)

    return res
#猜测字段
negatives_cczd = [
    '抱歉，没有符合的字段类型。',
    '对不起，找不到相应的字段类型。',
    '对不起，没有匹配的文档类型。',
    '很抱歉，相关的字段种类不存在。',
    '没有所符合的字段。',
    '没有匹配的字段存在。',
    '无法确定可能来源字段'
    '不能确定可能的来源字段。',
    '未能识别可能的来源文档。',
    '无法鉴别可能的来源文件。',
    '难以确定可能的字段出处。',
]
def reward_cczd_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        choices = ext['choice']
        answers = re.findall(r'“(.*?)”', completion)
        if len(answers) == 0:
            '''判断否定'''
            if completion not in negatives_cczd:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            '''判断是否都在choice中'''
            is_err = False
            for item in answers:
                if item not in choices:
                    rewards.append(0.0)
                    is_err = True
                    break
            if not is_err:
                rewards.append(1.0)
    return rewards

def reward_cczd_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in negatives_cczd and ext['answer'][0] == '否定':
            '''否定，直接判断是不是否定'''
            rewards.append(1.0)
        else:
            '''不否定，提取答案计算正确率'''
            answers = set(list(re.findall(r'“(.*?)”', completion)))
            corr = 0
            for ans in answers:
                if ans in ext['answer']:
                    corr += 1
            rewards.append(corr/len(ext['answer']))
    return rewards
#格式化


def reward_gsh_format(completions, target, extra_info):
    '''是否是json'''
    '''指标：可以解析成json=1 不能解析成json=0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            js_data = json.loads(completion)
            if '{' in completion and '}' in completion:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def reward_gsh_key(completions, target, extra_info):
    '''json的key正确性'''
    '''指标：召回率'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            pred = completion.strip()
            gold = gt.strip()
            gold_dict = json.loads(gold)
            pred_dict = json.loads(pred)

            preds = []
            golds = []
            
            for key, value in extract_key_value_pairs(gold_dict):
                golds.append(key)
            for key, value in extract_key_value_pairs(pred_dict):
                preds.append(key)

            gold_keys = set(golds)
            pred_keys = set(preds)
            
            corr_keys = pred_keys & gold_keys
            rewards.append(len(corr_keys)/ len(gold_keys))
        except:
            rewards.append(0.0)
    return rewards

def reward_gsh_value(completions, target, extra_info):
    '''答案是否正确，key-value对'''
    '''指标：f1'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            pred = completion.strip()
            gold = gt.strip()
            gold_dict = json.loads(gold)
            pred_dict = json.loads(pred)

            preds = []
            golds = []
            
            for key, value in extract_key_value_pairs(gold_dict):
                golds.append((key, value))
            for key, value in extract_key_value_pairs(pred_dict):
                preds.append((key, value))

            gold_tuples = set(golds)
            pred_tuples = set(preds)
            
            corr_tuples = pred_tuples & gold_tuples
            try:
                pre = len(corr_tuples) / len(pred_tuples)
                rec = len(corr_tuples) / len(gold_tuples)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
        except:
            rewards.append(0.0)
    return rewards
#科室导诊
ksdz_negatives = [
'根据该患者情况，给定选项中没有合适的科室。',
'无法在给定选项中找到符合该患者情况的科室。',
'根据该患者的病情，在提供的选项中没有适当的科室。',
'考虑到该患者的情况，所给科室选项中都不合适。',
'基于该患者的实际情况，选项中并未包含合适的科室。',
'该患者的情况与所给的科室选项不匹配。',
'在所提供的选项中，没有与该患者病情相匹配的科室。',
]
def reward_ksdz_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        choices = ext['choice']
        answers = []
        for choice in choices:
            if choice in completion:
                answers.append(choice)
        if len(answers) == 0:
            '''判断否定'''
            if completion not in ksdz_negatives:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            '''判断是否都在choice中'''
            is_err = False
            for item in answers:
                if item not in choices:
                    is_err = True
                    rewards.append(0.0)
                    break
            if not is_err:
                rewards.append(1.0)
    return rewards

def reward_ksdz_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in ksdz_negatives and ext['answer'][0] == '否定':
            '''否定，直接判断是不是否定'''
            rewards.append(1.0)
        else:
            '''不否定，提取答案计算正确率'''
            choices = ext['choice']
            answers = []
            for choice in choices:
                if choice in completion:
                    answers.append(choice)
            corr = 0
            for ans in answers:
                if ans in ext['answer']:
                    corr += 1
            rewards.append(corr/len(ext['answer']))
    return rewards
#指标提取
zbtq_negatives = [
'抱歉，没有符合该日期的检验信息',
'对不起，该日期的检验信息找不到。',
'很遗憾，没有找到该日期的检验记录。',
'对不起，该日期没有相关的检验数据。',
'很抱歉，未能查询到该日期的检验信息。',
'对不起，对应这个日期的检验信息缺失。'
]

def reward_zbtq_format(completions, target, extra_info):
    '''是否是json'''
    '''指标：可以解析成json=1 不能解析成json=0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in zbtq_negatives:
            completion = '{"答案":"否定"}'
        try:
            js_data = json.loads(completion)
            if '{' in completion and '}' in completion:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def reward_zbtq_hallucination(completions, target, extra_info):
    '''出现幻觉就是0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info):
      try:
        numbers = re.findall(r'[\d\.]+', completion)
        inp = ext['input']
        find_err = False
        for item in numbers:
            if item not in inp:
                print('reward_zbtq_hallucination')
                print('item not found:{}'.format(item))
                print('id:{}'.format(ext['id']))
                rewards.append(0.0)
                find_err = True
                break
        if not find_err:
            rewards.append(1.0)
      except Exception as e:
        rewards.append(0.0)
    return rewards


def reward_zbtq_key(completions, target, extra_info):
    '''json的key正确性'''
    '''指标：召回率'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in zbtq_negatives:
            completion = '{"答案":"否定"}'
        try:
            pred_json = json.loads(completion)
            pred_tuples, pred_keys = flatten_json_extract(pred_json)
            gold_keys = set(ext['keys'])
            corr_keys = pred_keys & gold_keys
            rewards.append(len(corr_keys)/ len(gold_keys))
        except:
            rewards.append(0.0)
    return rewards

def reward_zbtq_value(completions, target, extra_info):
    '''答案是否正确，key-value对'''
    '''指标：f1'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in zbtq_negatives:
            completion = '{"答案":"否定"}'
        try:
            pred_tuples = []
            gold_tuples = []
            pred_json = json.loads(completion)
            pred_tuples, pred_keys = flatten_json_extract(pred_json)
            gold_tuples = set(ext['tuples'])
            corr_tuples = pred_tuples & gold_tuples
            try:
                pre = len(corr_tuples) / len(pred_tuples)
                rec = len(corr_tuples) / len(gold_tuples)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
        except:
            rewards.append(0.0)
    return rewards
#指标检测
zbjc_positives = [
    '所有检验信息全部正确。',
    '找不到错误的检验信息。',
    '检验信息与文书完全匹配。',
    '检验信息与文书完全匹配。',
    '检验信息都是正确无误的。',
    '没有发现任何不准确的检验信息。',
    '所有的检验资料都是准确的。',
    '未发现有误的检验数据。'
]
def reward_zbjc_format(completions, target, extra_info):
    '''是否正确输出答案，或者是positive'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        answers = re.findall(r'在(.*?)进行的(.*?)检验存在数据不一致，医疗文本中记录的结果为(.*?)，而实际检验结果为(.*?)。', completion)
        if len(answers) == 0:
            '''判断否定'''
            if completion not in zbjc_positives:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(1.0)
    return rewards


def reward_zbjc_hallucination(completions, target, extra_info):
    '''出现幻觉就是0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        answers = re.findall(r'在(.*?)进行的(.*?)检验存在数据不一致，医疗文本中记录的结果为(.*?)，而实际检验结果为(.*?)。', completion)
        if len(answers) == 0:
            if completion not in zbjc_positives:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        else:
            inp = ext['input']
            find_err = False
            for item in answers:
                val_1 = item[2]
                if val_1 not in inp:
                    print('reward_zbjc_hallucination')
                    print('item not found:{}'.format(item))
                    print('id:{}'.format(ext['id']))
                    rewards.append(0.0)
                    find_err = True
                    break
                val_2 = item[3]
                if val_2 not in inp:
                    print('reward_zbjc_hallucination')
                    print('item not found:{}'.format(item))
                    print('id:{}'.format(ext['id']))
                    rewards.append(0.0)
                    find_err = True
                    break
            if not find_err:
                rewards.append(1.0)
    return rewards



def reward_zbjc_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if completion in zbjc_positives and ext['answer'][0] == '肯定':
            '''肯定，直接判断是不是肯定'''
            rewards.append(1.0)
        else:
            '''不肯定，提取答案计算正确率'''
            answers = re.findall(r'在(.*?)进行的(.*?)检验存在数据不一致，医疗文本中记录的结果为(.*?)，而实际检验结果为(.*?)。', completion)
            pred_tuples = []
            for item in answers:
                pred_tuples.append('||'.join(item))
            gold_tuples = ext['answer']
            gold_tuples = set(gold_tuples)
            pred_tuples = set(pred_tuples)
            corr_tuples = pred_tuples & gold_tuples
            try:
                pre = len(corr_tuples) / len(pred_tuples)
                rec = len(corr_tuples) / len(gold_tuples)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
    return rewards
# 出院小结
#患者基本信息
def reward_ds_pbi_format(completions, target, extra_info):
    '''是否是json'''
    '''指标：可以解析成json=1 不能解析成json=0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            js_data = json.loads(completion)
            rewards.append(1.0)
        except:
            rewards.append(0.0)
    return rewards

def reward_ds_pbi_key(completions, target, extra_info):
    '''json的key正确性'''
    '''指标：召回率'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            pred_json = json.loads(completion)
            pred_keys = set(list(pred_json.keys()))
            gold_keys = set(ext['keys'])
            corr_keys = pred_keys & gold_keys
            rewards.append(len(corr_keys)/ len(gold_keys))
        except:
            rewards.append(0.0)
    return rewards

def reward_ds_pbi_value(completions, target, extra_info):
    '''答案是否正确，key-value对'''
    '''指标：f1'''
    rewards = []
    threshold = 0.8
    mohu_keys = [
        '入院时简要病史',
        '体检摘要',
        '入院诊断'
    ]
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        try:
            all_keys = 0
            corr_keys = 0
            pred_keys = 0

            pred = completion.strip()
            gold = gt.strip()
            gold_dict = json.loads(gold)
            gold_dict = {k:translate_punctuation(str(v)) for k,v in gold_dict.items()}
            pred_dict = json.loads(pred)
            pred_dict = {k:translate_punctuation(str(v)) for k,v in pred_dict.items()}

            exact_pred_items = []
            exact_gold_items = []

            mohu_pred_items = {}
            mohu_gold_items = {}

            for key in gold_dict.keys():
                if key in mohu_keys:
                    mohu_gold_items[key] = gold_dict[key]
                else:
                    exact_gold_items.append((key,gold_dict[key]))

            for key in pred_dict.keys():
                if key in mohu_keys:
                    mohu_pred_items[key] = pred_dict[key]
                else:
                    exact_pred_items.append((key,pred_dict[key]))

            
            # 模糊的
            mohu_corr = []
            mohu_pred = []
            mohu_gold = []
            mohu_all = []

            mohu_keys = []
            mohu_keys.extend(list(mohu_pred_items.keys()))
            mohu_keys.extend(list(mohu_gold_items.keys()))
            mohu_keys = list(set(mohu_keys))

            for key in mohu_keys:
                if key in mohu_gold_items.keys() and key in mohu_pred_items.keys():
                    if mohu_gold_items[key].strip() == '':
                        continue
                    score = get_rouge_scores(mohu_pred_items[key], mohu_gold_items[key])
                    mohu_pred.append((key,mohu_pred_items[key]))
                    mohu_gold.append((key,mohu_gold_items[key]))
                    if score >= threshold:
                        mohu_corr.append((key,mohu_pred_items[key]))
                        mohu_all.append((key,mohu_pred_items[key]))
                    else:
                        mohu_all.append((key,mohu_pred_items[key]))
                        mohu_all.append((key,mohu_gold_items[key]))
                elif key in mohu_gold_items.keys():
                    if mohu_gold_items[key].strip() == '':
                        continue
                    mohu_gold.append((key,mohu_gold_items[key]))
                    mohu_all.append((key,mohu_gold_items[key]))
                elif key in mohu_pred_items.keys():
                    mohu_pred.append((key,mohu_pred_items[key]))
                    mohu_all.append((key,mohu_pred_items[key]))
                else:
                    raise ValueError('错误，请检查')

            # 全部匹配的
            exact_preds = set(exact_pred_items)
            exact_golds = set(exact_gold_items)
            exact_corr = exact_preds & exact_golds
            exact_all = exact_preds | exact_golds

            all_keys += len(mohu_gold)
            all_keys += len(exact_golds)

            corr_keys += len(mohu_corr)
            corr_keys += len(exact_corr)

            pred_keys += len(mohu_pred)
            pred_keys += len(exact_preds)
            try:
                p = corr_keys / pred_keys
                r = corr_keys / all_keys
                f1 = (2*p*r)/(p+r)
            except:
                f1 = 0
            rewards.append(f1)
        except:
            rewards.append(0.0)
    return rewards

#出院诊断
def reward_think_format(completions, target, extra_info):
    '''格式，要<think>xxxx</think>xxxx'''
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        completion = completion.strip()   
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>([^<]*(?:<(?!/?think>)[^<]*)*)$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def reward_think_value(completions, target, extra_info):
    '''匹配结果'''
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            completion = completion.strip()   
            match = re.search(r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>([^<]*(?:<(?!/?think>)[^<]*)*)$", completion)
            if match is None:
                rewards.append(0.0)
                continue

            ans = match.group(2).strip()
            ans = re.sub('\s','',ans)
            gt = re.sub('\s','',gt)
            if ans == gt:
                rewards.append(1.0) 
            else:
                rewards.append(0.0) 
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def reward_think_bertscore(completions, target, extra_info):
    '''匹配结果'''
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            ans = completion.strip()   
            p,r,f1 = get_bert_scores([ans], [gt])
            rewards.append(f1.item())
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def reward_think_rouge(completions, target, extra_info):
    '''匹配结果'''
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            ans = completion.strip()
            rewards.append(get_rouge_scores(ans, gt))
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards
def reward_think_value_dd(completions, target, extra_info):
    '''匹配结果'''
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            completion = completion.strip()   
            match = re.search(r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>([^<]*(?:<(?!/?think>)[^<]*)*)$", completion)
            if match is None:
                rewards.append(0.0)
                continue

            ans = match.group(2).strip()
            ans = re.sub('\s','',ans)
            gt = re.sub('\s','',gt)
            ans = ans.replace('，','')
            gt = gt.replace('，','')
            pred_dd = set(list(ans.split(',')))
            gold_dd = set(list(gt.split(',')))

            corr_dd = pred_dd & gold_dd
            try:
                pre = len(corr_dd) / len(pred_dd)
                rec = len(corr_dd) / len(gold_dd)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards
#病程与治疗情况
'''reward使用think_format以及think_value以及think_rouge'''
#出院时情况
'''reward使用think_format以及think_value以及think_rouge'''
#出院后用药建议
'''reward使用think_format以及think_value以及think_rouge'''
#住院期间医疗情况
def reward_ds_mr_hallucination(completions, target, extra_info):
    '''出现幻觉就是0'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info):
        completion = re.sub(r'[\(（].*?[\)）]','',completion)
        try:
            numbers = re.findall(r'[\d\.]+', completion)
            inp = ext['input']
            inp = re.sub('[×x·^]','', inp)
            find_err = False
            print('completion:{}'.format(completion))
            print('numbers:{}'.format(numbers))
            for item in numbers:
                if item not in inp:
                    print('reward_ds_mr_hallucination')
                    print('item not found:{}'.format(item))
                    print('id:{}'.format(ext['id']))
                    rewards.append(0.0)
                    find_err = True
                    break
            if not find_err:
                rewards.append(1.0)
        except Exception as e:
            print('reward_ds_mr_hallucination')
            print('item not found:{}'.format(item))
            print('id:{}'.format(ext['id']))
            rewards.append(0.0)
    return rewards

def reward_ds_mr_number(completions, target, extra_info):
    '''匹配数值'''
    rewards = []
    for completion, gt in zip(completions, target):
        completion = re.sub(r'[\(（].*?[\)）]','',completion)
        try:
            pred_numbers = set(list(re.findall(r'[\d\.]+', completion)))
            gold_numbers = set(list(re.findall(r'[\d\.]+', gt)))

            corr_numbers = pred_numbers & gold_numbers
            try:
                pre = len(corr_numbers) / len(pred_numbers)
                rec = len(corr_numbers) / len(gold_numbers)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards

def reward_ds_mr_name_and_number(completions, target, extra_info):
    '''名称+数值'''
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            pred_numbers = set(extract_tuples(completion))
            gold_numbers = set(extract_tuples(gt))

            corr_numbers = pred_numbers & gold_numbers
            try:
                pre = len(corr_numbers) / len(pred_numbers)
                rec = len(corr_numbers) / len(gold_numbers)
                f1 = 2*pre*rec / (pre+rec)
            except:
                f1 = 0
            rewards.append(f1)
        except Exception as e:
            print(e)
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards
############################################################################################

# 术语使用term的
term_negatives = [
    '无对应标准术语',
]
def reward_term_norm_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        choices = ext['choice']
        answers = re.findall(r'输出：(.*?)$', completion)
        if len(answers) == 1:
            answer_list = answers[0].split(';')
            is_err = False
            for answer in answer_list:
                if answer not in choices and answer not in term_negatives:
                    is_err = True
            if is_err:
                rewards.append(0.0)
            else:
                rewards.append(1.0)

        else:
            rewards.append(0.0)
    return rewards
overall_terms = []
with open('/HL_user01/2025-03-27-GRPO/2025-03-21-术语任务/全部术语.txt', 'r') as f:
    for line in f.readlines():
        if line.strip() == '':
            continue
        overall_terms.append(line.strip())

def reward_term_norm_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        answers = re.findall(r'输出：(.*?)$', completion)
        if len(answers) == 1:
            answer_list = answers[0].split(';')
            answer_list = [item for item in answer_list if item != '']
            answer = set(answer_list)
            gold = set(ext['answer'])
            corr = answer & gold
            try:
                p = len(corr) / len(answer)
                r = len(corr) / len(gold)
                f1 = 2*p*r / (p+r)
            except:
                p = 0
                r = 0
                f1 = 0
            rewards.append(f1)
        else:
            rewards.append(0.0)
    return rewards
def reward_term_isa_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        is_err = False
        for item in completion.strip().split('，'):
            if item not in overall_terms:
                is_err = True
        if is_err:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

def reward_term_isa_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        answer_list = completion.split('，')
        answer = set(answer_list)
        gold = set(gt.split('，'))
        corr = answer & gold
        try:
            p = len(corr) / len(answer)
            r = len(corr) / len(gold)
            f1 = 2*p*r / (p+r)
        except:
            p = 0
            r = 0
            f1 = 0
        rewards.append(f1)
    return rewards

def reward_term_syn_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        is_err = False
        for item in completion.strip().split('，'):
            if item not in overall_terms:
                is_err = True
        if is_err:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

def reward_term_syn_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        answer_list = completion.split('，')
        answer = set(answer_list)
        gold = set(gt.split('，'))
        corr = answer & gold
        try:
            p = len(corr) / len(answer)
            r = len(corr) / len(gold)
            f1 = 2*p*r / (p+r)
        except:
            p = 0
            r = 0
            f1 = 0
        rewards.append(f1)
    return rewards

def reward_term_link_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info):
        split_index = completion.index('综上所述') 
        process_completion = completion[:split_index]
        process_entities = ext['process_entities']
        contains_num = 0
        for entity in process_entities:
            if entity in process_completion:
                contains_num += 1
        rewards.append(contains_num/len(process_entities))
    return rewards

def reward_term_link_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        split_index = completion.index('综上所述') 
        ans_completion = completion[split_index:]
        ans_entities = ext['step_golden']
        contains_num = 0
        for entity in ans_entities:
            if entity in ans_completion:
                contains_num += 1
        rewards.append(contains_num/len(ans_entities))
    return rewards

pattern_format_qcic = r'### 思考过程：(.*?)### 最终答案：(?:Supported|Not Supported|Not Sure)'
pattern_answer_qcic = r'\b(Supported|Not Supported|Not Sure)\b'

def extract_status(text):
    if not isinstance(text, (str, bytes)):
        text = str(text) if text is not None else ""
    pattern = r'\b(Supported|Not Supported|Not Sure)\b'
    matches = re.findall(pattern, text)
    if matches:
        return "Yes" if matches[-1] == "Supported" else "No"
    return "No"

def reward_qcic_format(completions, target, extra_info):
    '''是否都在答案中或者是否是negative'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        completion = completion.strip()
        if re.fullmatch(pattern_format_qcic,completion,flags=re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_qcic_value(completions, target, extra_info):
    '''答案是否正确'''
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        try:
            ans = extract_status(completion)
            if ans == ext['label']:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

############################################################################################

# 最终的reward函数
def reward_instruction_emr(completions, target, extra_info, **kwargs):
    rewards = []
    for completion, gt, ext in zip(completions, target, extra_info): 
        try:
            if ext['task'] == 'cczd':
                reward_1 = reward_cczd_format([completion], [gt], [ext])
                reward_2 = reward_cczd_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'gsh':
                reward_1 = reward_gsh_format([completion], [gt], [ext])
                reward_2 = reward_gsh_key([completion], [gt], [ext])
                reward_3 = reward_gsh_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(reward_1[0]+reward_2[0]+2*reward_3[0])
            elif ext['task'] == 'ksdz':
                reward_1 = reward_ksdz_format([completion], [gt], [ext])
                reward_2 = reward_ksdz_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'zbtq':
                reward_1 = reward_zbtq_format([completion], [gt], [ext])
                reward_2 = reward_zbtq_hallucination([completion], [gt], [ext])
                reward_3 = reward_zbtq_key([completion], [gt], [ext])
                reward_4 = reward_zbtq_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                assert len(reward_4) == 1, print('{} reward_4 len error:{}'.format(ext['task'], reward_4))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                assert reward_4[0]<=1.0, print('{} reward_4 value error:{}'.format(ext['task'], reward_4))
                rewards.append(reward_1[0]+reward_2[0]+reward_3[0]+reward_4[0])
            elif ext['task'] == 'zbjc':
                reward_1 = reward_zbjc_format([completion], [gt], [ext])
                reward_2 = reward_zbjc_hallucination([completion], [gt], [ext])
                reward_3 = reward_zbjc_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(reward_1[0]+1.5*reward_2[0]+1.5*reward_3[0])
            elif ext['task'] == 'cyxj_pbi':
                reward_1 = reward_ds_pbi_format([completion], [gt], [ext])
                reward_2 = reward_ds_pbi_key([completion], [gt], [ext])
                reward_3 = reward_ds_pbi_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(reward_1[0]+1.5*reward_2[0]+1.5*reward_3[0])
            elif ext['task'] == 'cyxj_dd':
                # reward_1 = reward_think_format([completion], [gt], [ext])
                reward_2 = reward_think_bertscore([completion], [gt], [ext])
                reward_3 = reward_think_rouge([completion], [gt], [ext])
                # assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                # assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(2*reward_2[0]+2*reward_3[0])
            elif ext['task'] == 'cyxj_hc':
                # reward_1 = reward_think_format([completion], [gt], [ext])
                reward_2 = reward_think_bertscore([completion], [gt], [ext])
                reward_3 = reward_think_rouge([completion], [gt], [ext])
                # assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                # assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(2*reward_2[0]+2*reward_3[0])
            elif ext['task'] == 'cyxj_ds':
                # reward_1 = reward_think_format([completion], [gt], [ext])
                reward_2 = reward_think_bertscore([completion], [gt], [ext])
                reward_3 = reward_think_rouge([completion], [gt], [ext])
                # assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                # assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(2*reward_2[0]+2*reward_3[0])
            elif ext['task'] == 'cyxj_rm':
                # reward_1 = reward_think_format([completion], [gt], [ext])
                reward_2 = reward_think_bertscore([completion], [gt], [ext])
                reward_3 = reward_think_rouge([completion], [gt], [ext])
                # assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                # assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(2*reward_2[0]+2*reward_3[0])
            elif ext['task'] == 'cyxj_ir':
                reward_1 = reward_ds_mr_hallucination([completion], [gt], [ext])
                reward_2 = reward_ds_mr_number([completion], [gt], [ext])
                reward_3 = reward_ds_mr_name_and_number([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert len(reward_3) == 1, print('{} reward_3 len error:{}'.format(ext['task'], reward_3))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                assert reward_3[0]<=1.0, print('{} reward_3 value error:{}'.format(ext['task'], reward_3))
                rewards.append(1.5*reward_1[0]+reward_2[0]+1.5*reward_3[0])
            elif ext['task'] == 'term_norm':
                reward_1 = reward_term_norm_format([completion], [gt], [ext])
                reward_2 = reward_term_norm_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'term_isa':
                reward_1 = reward_term_isa_format([completion], [gt], [ext])
                reward_2 = reward_term_isa_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'term_syn':
                reward_1 = reward_term_syn_format([completion], [gt], [ext])
                reward_2 = reward_term_syn_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'term_link':
                reward_1 = reward_term_link_format([completion], [gt], [ext])
                reward_2 = reward_term_link_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            elif ext['task'] == 'qcic_opensource' or ext['task'] == 'qcic_hospital':
                reward_1 = reward_qcic_format([completion], [gt], [ext])
                reward_2 = reward_qcic_value([completion], [gt], [ext])
                assert len(reward_1) == 1, print('{} reward_1 len error:{}'.format(ext['task'], reward_1))
                assert len(reward_2) == 1, print('{} reward_2 len error:{}'.format(ext['task'], reward_2))
                assert reward_1[0]<=1.0, print('{} reward_1 value error:{}'.format(ext['task'], reward_1))
                assert reward_2[0]<=1.0, print('{} reward_2 value error:{}'.format(ext['task'], reward_2))
                rewards.append(2*reward_1[0]+2*reward_2[0])
            else:
                print('task error:{}'.format(ext['task']))
                rewards.append(0.)
        except Exception as e:
            print('!!!error:{}'.format(e))
            rewards.append(0.)
        print('#######################################################')
        print('completion:{}'.format(completion))
        print('gt:{}'.format(gt))
        print('rewards:{}'.format(rewards))

    return rewards

############################################################################################

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    '''若重新安装 trl，务必在库中的代码加上add_special_tokens'''
    special_tokens = ["<|user|>", "<|assistant|>","[gMASK]","sop"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    test_messages = [
        {"role":"user", "content":"你好"}
    ]
    print('测试数据')
    print(tokenizer.apply_chat_template(test_messages,add_generation_prompt=True, tokenize=False))
    print(tokenizer.apply_chat_template(test_messages,add_generation_prompt=True, tokenize=True))
    print('dataset:{}'.format(script_args.dataset_id_or_path))
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    with jsonlines.open(script_args.dataset_id_or_path, 'r') as f:
        datas = [data for data in f]
    dataset = Dataset.from_list(datas)

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(ins, output):
        # r1_prefix = [{
        #     "role": "user",
        #     "content": ins
        #   }]
        # chatglm3直接强制加上即可
        # prompt = "<|user|>\n {}\n<|assistant|>".format(ins.strip())
        # prompt = ins.strip()

        messages = [{"role": "user", "content": ins}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"prompt": prompt,'target': output}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["instruction"], x['output']))
    print('generate r1后:{}'.format(dataset[0]))
    dataset.shuffle()
    # split the dataset into train and test
    # train_test_split = dataset.train_test_split(test_size=0.1)

    # train_dataset = train_test_split["train"]
    # test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[reward_instruction_emr],
      args=training_args,
      train_dataset=dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()