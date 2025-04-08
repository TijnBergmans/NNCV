wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 100 \
    --lr 6e-5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Mask2Former (pre)training" \
    --checkpoint-interval 5 \
    --early-stopping-patience 10 \
    --augmentation "lsj" \
    --pre-train True \