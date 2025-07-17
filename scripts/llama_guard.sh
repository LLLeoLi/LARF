MODEL_NAME="/mnt/petrelfs/share_data/safety_verifier/Llama-Guard-3-8B"
srun -p AI4Good_L1_p --gres=gpu:1 -J eval vllm serve $MODEL_NAME --dtype auto --api-key token-pjlab --port 8612