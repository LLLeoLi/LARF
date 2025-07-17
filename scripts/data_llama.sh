start_begin=10
end_begin=11

for (( begin_num=$start_begin; begin_num<=$end_begin; begin_num++ ))
do
    end_num=$((begin_num + 1))
    SCRIPT_FILE="temp_script_${begin_num}_${end_num}.sh"
    cat > $SCRIPT_FILE <<EOF
#!/bin/bash
#SBATCH --job-name=${begin_num}-${end_num}
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
#SBATCH --output=llama_sort_results/llama-${begin_num}-${end_num}.log
python get_bi_rep_llama.py --layer_num_start $begin_num --layer_num_end $end_num
EOF
    
    echo "Running with begin_num=$begin_num, end_num=$end_num"
    sbatch $SCRIPT_FILE
    rm $SCRIPT_FILE
    sleep 0.05
done