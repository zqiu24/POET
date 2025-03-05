. /home/zqiu/anaconda3/etc/profile.d/conda.sh
conda activate galore

export WANDB_API_KEY="254b491166b72b5f961613863d702748580bead9"

# Get the index argument
idx=$1

# If idx > 6, exit
# if [ $idx -gt 6 ]; then
#     echo "Index $idx is greater than 6, exiting..."
#     exit 0
# fi

# Choose optimizer based on idx
if [ $idx -eq 0 ]; then
    optimizer="galore_adamw"
    lr=0.01
elif [ $idx -eq 1 ]; then
    optimizer="adamw_baseline"
    lr=0.001
elif [ $idx -eq 2 ]; then
    optimizer="soft_adamw"
    lr=0.001
elif [ $idx -eq 3 ]; then
    optimizer="soft_adamw_neumann"
    lr=0.001
fi

echo "Optimizer: $optimizer"

# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr $lr \
    --galore_scale 0.25 \
    --rank 128 \
    --soft_rank 4 \
    --update_proj_gap 200 \
    --batch_size 512 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --grad_clipping 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer $optimizer \
    --reset_R \

#     --scheduler constant
#     --batch_size 256 \
#     --warmup_steps 1000 \
#     --num_training_steps 10000 \ 

python hold.py