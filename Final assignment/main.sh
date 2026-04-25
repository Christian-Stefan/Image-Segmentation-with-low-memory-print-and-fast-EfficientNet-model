wandb login
python3 train_hypo.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 20 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Hyperparameter Optimization with the old set-up" \
