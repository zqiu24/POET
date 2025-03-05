. /home/zqiu/anaconda3/etc/profile.d/conda.sh
conda activate galore

export WANDB_API_KEY="254b491166b72b5f961613863d702748580bead9"

# Get the index argument
idx=$1

# If idx > 3, exit
if [ $idx -gt 3 ]; then
    echo "Index $idx is greater than 3, exiting..."
    exit 0
fi

# Choose optimizer based on idx
if [ $idx -eq 0 ]; then
    optimizer="galore_adamw"
elif [ $idx -eq 1 ]; then
    optimizer="galore_adamw8bit"
elif [ $idx -eq 2 ]; then
    optimizer="adam"
else
    optimizer="adam8bit"
fi

echo "Optimizer: $optimizer"

# LLaMA-130M, GaLore-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer $optimizer