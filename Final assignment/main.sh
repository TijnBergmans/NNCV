wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 24 \
    --epochs 150 \
    --lr 6e-5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "(Pre) Training Half-res" \
    --checkpoint-interval 5 \
    --early-stopping-patience 10 \
    --augmentation "lsj" \
    --pre-train 0 \