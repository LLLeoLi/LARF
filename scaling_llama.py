import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import copy
import pandas as pd
import argparse
from tqdm import tqdm


def scaling(base_model, begin_num, end_num, scale_num):
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            new_model.model.layers[i].self_attn.q_proj.weight.copy_(\
                base_model.model.layers[i].self_attn.q_proj.weight*scale_num)
            new_model.model.layers[i].self_attn.k_proj.weight.copy_(\
                base_model.model.layers[i].self_attn.k_proj.weight*scale_num)
            new_model.model.layers[i].self_attn.v_proj.weight.copy_(\
                base_model.model.layers[i].self_attn.v_proj.weight*scale_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(\
                base_model.model.layers[i].self_attn.o_proj.weight*scale_num)
            new_model.model.layers[i].mlp.up_proj.weight.copy_(\
                base_model.model.layers[i].mlp.up_proj.weight*scale_num)
            new_model.model.layers[i].mlp.gate_proj.weight.copy_(\
                base_model.model.layers[i].mlp.gate_proj.weight*scale_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(\
                base_model.model.layers[i].mlp.down_proj.weight*scale_num)
    return new_model


base_model = AutoModelForCausalLM.from_pretrained('/mnt/petrelfs/lihao1/models/meta-llama/llama3-8b-instruct', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/lihao1/models/meta-llama/llama3-8b-instruct')
tokenizer.pad_token = tokenizer.eos_token

outputs = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin_num', type=int, default=16)
    parser.add_argument('--end_num', type=int, default=17)
    parser.add_argument('--scale_num', type=float, default=1.1)
    args = parser.parse_args()
    begin_num = args.begin_num
    end_num = args.end_num
    scale_num = args.scale_num
    query_list = json.load(open('data/overrejection_final.json','r'))
    model = scaling(base_model, begin_num, end_num, scale_num)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    for query in tqdm(query_list):
        messages = [
            {
                'role':'user',
                'content':query,
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids = tokenizer([text], return_tensors="pt").to(model.device)
        model.eval()
        generation = model.generate(
            **input_ids,
            max_new_tokens=32,
            do_sample=False
        )
        res = tokenizer.decode(generation[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        outputs.append({
            "instruction": query,
            "output": res
        })
    with open(f'scaling/llama-{begin_num}-{end_num}-{scale_num}-overrejection_final.json','w') as f:
         json.dump(outputs, f, indent=4)