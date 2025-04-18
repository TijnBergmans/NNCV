wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 500 \
    --lr 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "(Pre) Training Half-res" \
    --early-stopping-patience 10 \
    --pre-train 0 \
    --load-weight 0\