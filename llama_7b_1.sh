. /home/wliu/anaconda3/etc/profile.d/conda.sh
conda activate galore

# MASTER_ADDR="172.22.8.7"

# LLaMA-7B, GaLore-Adam, 8 A100, 8 Node
# Run this on the master node (NODE_RANK=0)
# FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 torchrun torchrun_main.py \
#     --model_config configs/llama_7b.json \
#     --lr 0.005 \
#     --galore_scale 0.25 \
#     --rank 1024 \
#     --update_proj_gap 500 \
#     --batch_size 8 \
#     --total_batch_size 512 \
#     --num_training_steps 15000 \
#     --warmup_steps 1500 \
#     --weight_decay 0 \
#     --grad_clipping 1.0 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --optimizer galore_adamw 


    # --num_training_steps 150000 \
    # --warmup_steps 15000 \

# -------------------------------------------------

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=172.22.8.7 \
    --master_port=29500 \
    torchrun_main.py \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer galore_adamw 

# torchrun --nnodes=2 --nproc_per_node=8 torchrun_main.py \
#     --model_config configs/llama_7b.json \
#     --lr 0.005 \
#     --galore_scale 0.25 \
#     --rank 1024 \
#     --update_proj_gap 500 \
#     --batch_size 8 \
#     --total_batch_size 512 \
#     --num_training_steps 150000 \
#     --warmup_steps 15000 \
#     --weight_decay 0 \
#     --grad_clipping 1.0 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --optimizer galore_adamw 