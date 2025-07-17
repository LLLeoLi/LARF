import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument('--layer_num_start', type=int, default=14)
parser.add_argument('--layer_num_end', type=int, default=15)
args = parser.parse_args()
layer_num_start = args.layer_num_start
layer_num_end = args.layer_num_end
avg = 100
select_n = 1000
file_name = 'alpaca'
res_logits_str = f'llama_bi_res_{layer_num_start}2{layer_num_end}_{file_name}_avg{avg}'
print(res_logits_str)
model_id = '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'
dir_id = 'llama_sort_results'
if not os.path.exists(dir_id):
    os.makedirs(dir_id)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             output_hidden_states=True,
                                             return_dict_in_generate=True,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

with open(f'data/train_data/alpaca-gpt4.json','r') as f:
    data = []
    target_data = json.load(f)
    
with open('data/pure-bad-100.jsonl', 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    unsafe_data = data

with open('data/pure-bad-100-anchor1.jsonl', 'r') as f:
    safe_data = []
    for line in f:
        safe_data.append(json.loads(line))

def alpaca_data_process(data):
    if 'input' in data.keys() and data['input'] != '':
        message = [
            {"role": "user", "content": f"{data['instruction']}\n{data['input']}"},
            {"role": "assistant", "content": data['output']},
        ]
    else:
        message = [
            {"role": "user", "content": f"{data['instruction']}"},
            {"role": "assistant", "content": data['output']},
        ]
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            message,
            return_tensors="pt"
        ).to(model.device)
        rep = model(input_ids).hidden_states[layer_num_start][0][-1].detach().cpu()
    return rep

def safety_data_process(data):
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            data['messages'],
            return_tensors="pt"
        ).to(model.device)
        rep = model(input_ids).hidden_states[layer_num_start][0][-1].detach().cpu()
    return rep

def top_cosine_similarity(A, B, C):
    A = A.to(torch.float32).numpy()
    B = B.to(torch.float32).numpy()
    C = C.to(torch.float32).numpy()
    unsafe_sim = cosine_similarity(A,B)
    safe_sim = cosine_similarity(A,C)
    final_sim = unsafe_sim - safe_sim
    sorted_similarity_matrix = np.sort(final_sim, axis=1)[:, ::-1]
    scores = np.mean(sorted_similarity_matrix[:, :avg], axis=1)
    rankings = np.argsort(scores)[::-1]
    return rankings, scores[rankings]

unsafe_rep = torch.stack([safety_data_process(data) for data in tqdm(unsafe_data, desc = 'unsafe_datas')])

safe_rep = torch.stack([safety_data_process(data) for data in tqdm(safe_data, desc = 'safe_datas')])

target_rep = torch.stack([alpaca_data_process(data) for data in tqdm(target_data , desc=f'target_datas:{file_name}')])

def process_and_save_data(indices, scores, target_data, output_path):
    values = []
    for i, s in zip(indices, scores):
        item = target_data[i].copy()
        item.update({'sim_score': float(s)})
        values.append(item)
    
    with open(output_path, 'w') as file:
        json.dump(values, file, indent=4)

indices, scores = top_cosine_similarity(target_rep, unsafe_rep, safe_rep)

process_and_save_data(indices, scores, target_data, f"{dir_id}/{res_logits_str}_mean.json")

top_indices = indices[:select_n]
top_scores = scores[:select_n]
process_and_save_data(top_indices, top_scores, target_data, f"{dir_id}/{res_logits_str}_mean_top_{select_n}.json")

bottom_indices = indices[-select_n:]
bottom_scores = scores[-select_n:]
process_and_save_data(bottom_indices, bottom_scores, target_data, f"{dir_id}/{res_logits_str}_mean_bottom_{select_n}.json")