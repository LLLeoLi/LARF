# LARF: Layer-Aware Representation Filtering

## Overview

This method identifies safety-sensitive layers within the LLM and leverages data representa tions to detect safety-degrading data samples in the fine-tuning dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/LLLeoLi/LARF.git
    cd LARF
    ```

2. Install dependencies:
    ```bash
    conda create --name LARF python=3.10
    conda activate LARF
    pip install -r requirements.txt
    ```

## Usage

### 1. Identify the Safety-Sensitive Layers
```bash
bash scripts/scaling_llama.sh
```

### 2. Filter Your Dataset with LARF
```bash
python get_bi_rep_llama.py --layer_num_start 10 --layer_num_end 11
```

### 3. Fine-tuning with LoRA
Train the model with custom datasets:
```bash
bash scripts/train_multipule_llama.sh
```

### 4. Safety Evaluation
Evaluate model outputs for safety:
```bash
bash scripts/llama_guard.sh
python llama_guard.py
```

### 5. Analysis
Open the Jupyter notebook for analysis:
```bash
jupyter notebook analysis.ipynb
```