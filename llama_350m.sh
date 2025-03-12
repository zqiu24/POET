. /home/wliu/anaconda3/etc/profile.d/conda.sh
conda activate galore

export WANDB_API_KEY="254b491166b72b5f961613863d702748580bead9"

# Get the index and GPU arguments
idx=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# gpu_id=$2  # New parameter for GPU ID

# Set the visible GPUs for this process based on the gpu_id
# if [ "$gpu_id" -eq 0 ]; then
#     export CUDA_VISIBLE_DEVICES=0,1,2,3
# elif [ "$gpu_id" -eq 1 ]; then
#     export CUDA_VISIBLE_DEVICES=4,5,6,7
# else
#     echo "Error: gpu_id must be 0 or 1"
#     exit 1
# fi

# Choose optimizer based on idx
if [ $idx -eq 0 ]; then
    optimizer="adamw"
    lr=0.0005 # 0.01, 0.005, 0.001, 0.0005, 0.0001
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
elif [ $idx -eq 1 ]; then
    optimizer="galore_adamw"
    lr=0.01
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
elif [ $idx -eq 2 ]; then
    optimizer="soft_adamw_neumann"
    lr=0.01
    soft_rank=4
    # Set soft_lr and update_reset_R_gap based on soft_rank
    if [ $soft_rank -eq 4 ]; then
        soft_lr_values=(0.002)
        update_reset_R_gap=200
    elif [ $soft_rank -eq 2 ]; then
        soft_lr_values=(0.001)
        update_reset_R_gap=200
    fi
elif [ $idx -eq 3 ]; then
    optimizer="only_adamw"
    lr=0.0005 # 0.01, 0.005, 0.001, 0.0005, 0.0001
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
elif [ $idx -eq 4 ]; then
    optimizer="only_galore_adamw"
    lr=0.01
    soft_lr_values=(0)
    soft_rank=0
    update_reset_R_gap=0
elif [ $idx -eq 5 ]; then
    optimizer="only_soft_adamw_neumann"
    lr=0.01
    soft_rank=4  # or 2
    # Set soft_lr and update_reset_R_gap based on soft_rank
    if [ $soft_rank -eq 4 ]; then
        soft_lr_values=(0.002)
        update_reset_R_gap=200
    elif [ $soft_rank -eq 2 ]; then
        soft_lr_values=(0.001)
        update_reset_R_gap=200
    fi
fi

# Loop through each training steps value
for soft_lr in "${soft_lr_values[@]}"; do
    echo "Running with optimizer: $optimizer, lr: $lr, soft_lr: $soft_lr, soft_rank: $soft_rank, update_reset_R_gap: $update_reset_R_gap"

    wandb_project="soft-paper-350m"
    
    # Add "-linear" suffix for indices outside 0, 1, 2
    if [ $idx -gt 2 ]; then
        wandb_project="${wandb_project}-linear"
    fi

    # LLaMA-350M, GaLore-Adam, 4 A100, 1 Node
    torchrun --standalone --nproc_per_node 8 torchrun_main.py \
        --wandb_project $wandb_project \
        --model_config configs/llama_350m.json \
        --lr $lr \
        --soft_lr $soft_lr \
        --galore_scale 0.25 \
        --rank 256 \
        --soft_rank $soft_rank \
        --soft_num_neumann_terms 5 \
        --update_proj_gap 200 \
        --update_reset_R_gap $update_reset_R_gap \
        --batch_size 64 \
        --total_batch_size 512 \
        --num_training_steps 300000 \
        --warmup_steps 0 \
        --min_lr_ratio 0.1 \
        --weight_decay 0 \
        --grad_clipping 0.1 \
        --dtype bfloat16 \
        --eval_every 10000 \
        --save_every 10000000 \
        --optimizer $optimizer \
        --reset_R \
        --stochastic \

done

#         --num_training_steps 60000 \