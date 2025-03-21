. /home/zqiu/anaconda3/etc/profile.d/conda.sh
conda activate galore

# Get the index argument
idx=$1

# If idx is not 0, exit
if [ $idx -ne 0 ]; then
    echo "Index $idx is not 0, exiting..."
    exit 0
fi

# LLaMA-7B, 8-bit GaLore-Adam, single GPU
# 22.72G, 0.37s/it
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 1 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer