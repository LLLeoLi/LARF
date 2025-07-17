files=()
for i in {10..11}; do
    j=$((i+1))
    files+=("llama_bi_res_${i}2${j}_alpaca_avg100_mean")
done
data_dir="llama_sort_results"
output_dir="llama_output"
mkdir -p $output_dir
for file in "${files[@]}"; do
    echo "processing ${file}_bottom_1000"
    SCRIPT_FILE="temp_script_llama_top.sh"
    cat > $SCRIPT_FILE <<EOF
#!/bin/bash
#SBATCH --job-name=${file}_bottom_1000
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_bottom_1000.log
python train_llama.py --data_path ${data_dir}/${file}_bottom_1000.json --output_path ${output_dir}/${file}_bottom_1000
EOF
    sbatch $SCRIPT_FILE
    rm $SCRIPT_FILE
    echo "processing ${file}_top_1000"
    SCRIPT_FILE="temp_script_llama_bottom.sh"
    cat > $SCRIPT_FILE <<EOF
#!/bin/bash
#SBATCH --job-name=${file}_top_1000
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
#SBATCH --output=${output_dir}/${file}_top_1000.log
python train_llama.py --data_path ${data_dir}/${file}_top_1000.json --output_path ${output_dir}/${file}_top_1000
EOF
    sbatch $SCRIPT_FILE
    rm $SCRIPT_FILE
done