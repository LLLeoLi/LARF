import os
import gc
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from peft import  LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer
)

model_id = '/mnt/petrelfs/lihao1/models/meta-llama/Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.enable_input_require_grads()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


def alpaca_process_func(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    if example['input'] == '':
        message = [
            {"role": "user", "content": f"{example['instruction']}"},
        ]
    else:
        message = [
            {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
        ]
    insturction = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = insturction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = insturction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(insturction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_goals(bench):
    goals = {
        'direct':pd.read_csv('safe_test/directHarm4.csv')['Goal'].to_list(),
        'harm':pd.read_csv('safe_test/harmbench.csv')['Goal'].to_list(),
        'phi':pd.read_csv('safe_test/phi.csv')['Goal'].to_list()
    }
    return goals[bench]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='llama_sort_results/llama_bi_res_logits_avg100_mean_bottom_1000.json')
    parser.add_argument('--output_path', type=str, default='llama_output/llama_bi_res_logits_avg100_mean_bottom_1000')
    args = parser.parse_args()
    train_json_path = args.data_path
    train_ds = Dataset.from_json(train_json_path)
    train_dataset = train_ds.map(alpaca_process_func)
    # LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        r=8,
    )
    peft_model = get_peft_model(model, config)

    # Trainer
    os.environ["WANDB_PROJECT"]="Llama"
    args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=1,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="wandb",
        warmup_ratio=0.1
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()
    
    del peft_model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    bench_list = ['direct','harm','phi']
    peft_id = train_json_path.split('/')[-1].split('.')[0]
    model = AutoModelForCausalLM.from_pretrained(f"llama_output/{peft_id}/checkpoint-186", device_map="auto")
    for bench in bench_list:
        goals = get_goals(bench)
        final_list = []
        for goal in tqdm(goals):
            messages = [
                {
                    'role':'user',
                    'content':goal,
                }
            ]
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            input_ids = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **input_ids,
                max_new_tokens=160,
                do_sample=False,
            )
            response = tokenizer.decode(outputs[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
            final_list.append({
                'instruction':goal,
                'output':response
            })
            with open(f'safe_test/results/{peft_id}-{bench}.json','w') as f:
                json.dump(final_list, f, indent=4)