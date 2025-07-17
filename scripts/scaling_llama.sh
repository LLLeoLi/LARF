start_begin=10
end_begin=11

scale_nums=(0.7 0.8 0.9 1.1 1.2 1.3)

for (( begin_num=$start_begin; begin_num<=$end_begin; begin_num++ ))
do
    end_num=$((begin_num + 1))
    
    for scale_num in "${scale_nums[@]}"
    do
        SCRIPT_FILE="temp_script_scaling_llama3_${begin_num}_${end_num}.sh"
        cat > $SCRIPT_FILE <<EOF
#!/bin/bash
#SBATCH --job-name=${begin_num}-${end_num}-${scale_num}
#SBATCH --partition=AI4Good_L1_p
#SBATCH --gres=gpu:1
#SBATCH --output=scaling/llama-${begin_num}-${end_num}-${scale_num}_final.log
python scaling_llama.py --begin_num $begin_num --end_num $end_num --scale_num $scale_num
EOF
        echo "Running with begin_num=$begin_num, end_num=$end_num, scale_num=$scale_num"
        sbatch $SCRIPT_FILE
        rm $SCRIPT_FILE
        sleep 0.05
    done
done