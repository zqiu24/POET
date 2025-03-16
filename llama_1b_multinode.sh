. /home/wliu/anaconda3/etc/profile.d/conda.sh
conda activate galore

export WANDB_API_KEY="254b491166b72b5f961613863d702748580bead9"

# Get the index and GPU arguments
idx=$1
node=$2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Choose optimizer based on idx
if [ $idx -eq 0 ]; then
    optimizer="adamw"
    lr=0.0005 # 0.01, 0.005, 0.001, 0.0005, 0.0001
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
    min_lr_ratio=0.1
elif [ $idx -eq 1 ]; then
    optimizer="galore_adamw"
    lr=0.01
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
    min_lr_ratio=0.1
elif [ $idx -eq 2 ]; then
    optimizer="soft_adamw_neumann"
    lr=0.005
    soft_rank=4
    # Set soft_lr and update_reset_R_gap based on soft_rank
    if [ $soft_rank -eq 4 ]; then
        soft_lr_values=(0.002)
        update_reset_R_gap=200
    elif [ $soft_rank -eq 2 ]; then
        soft_lr_values=(0.001)
        update_reset_R_gap=200
    fi
    min_lr_ratio=0.01
elif [ $idx -eq 3 ]; then
    optimizer="only_adamw"
    lr=0.0005 # 0.01, 0.005, 0.001, 0.0005, 0.0001
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
    min_lr_ratio=0.1
elif [ $idx -eq 4 ]; then
    optimizer="only_galore_adamw"
    lr=0.01
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
    min_lr_ratio=0.1
elif [ $idx -eq 5 ]; then
    optimizer="only_soft_adamw_neumann"
    lr=0.01
    soft_rank=4
    # Set soft_lr and update_reset_R_gap based on soft_rank
    if [ $soft_rank -eq 4 ]; then
        soft_lr_values=(0.002)
        update_reset_R_gap=100
    elif [ $soft_rank -eq 2 ]; then
        soft_lr_values=(0.001)
        update_reset_R_gap=200
    fi
    min_lr_ratio=0.01
fi

# Loop through each training steps value
for soft_lr in "${soft_lr_values[@]}"; do
    echo "Running with optimizer: $optimizer, lr: $lr, soft_lr: $soft_lr, soft_rank: $soft_rank, update_reset_R_gap: $update_reset_R_gap"

    wandb_project="soft-paper-1b-benchmark"
    
    # Add "-linear" suffix for indices outside 0, 1, 2
    if [ $idx -gt 2 ]; then
        wandb_project="${wandb_project}-linear"
    fi

    # LLaMA-1B, GaLore-Adam, 8 A100, 1 Node
    torchrun \
        --nnodes=8 \
        --nproc_per_node=8 \
        --node_rank=$node \
        --master_addr=172.22.8.9 \
        --master_port=29500 \
        torchrun_main.py \
        --wandb_project $wandb_project \
        --model_config configs/llama_1b.json \
        --lr $lr \
        --soft_lr $soft_lr \
        --galore_scale 0.25 \
        --rank 512 \
        --soft_rank $soft_rank \
        --soft_num_neumann_terms 5 \
        --update_proj_gap 200 \
        --update_reset_R_gap $update_reset_R_gap \
        --batch_size 64 \
        --total_batch_size 4096 \
        --num_training_steps 1250 \
        --warmup_steps 0 \
        --min_lr_ratio $min_lr_ratio \
        --weight_decay 0 \
        --grad_clipping 0.01 \
        --dtype bfloat16 \
        --eval_every 10000 \
        --save_every 100000000 \
        --optimizer $optimizer \
        --reset_R \
        --stochastic \

done
    # --batch_size 16 \
    #     --num_training_steps 100000 \
    #     --warmup_steps 10000 \