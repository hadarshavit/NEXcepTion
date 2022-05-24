Training code for NEXcepTion.


For NEXcepTion-T run:
```
python -m torch.distributed.launch --nproc_per_node=4 train.py DATA_PATH --amp --batch-size 64 --opt fusedlamb --weight-decay 0.02 --sched 'cosine' --input-size 3 224 224 --drop-path 0.05 --channels-last -j 8 --bce-loss --bce-target-thresh 0.2 --epochs 300  --weight-decay 0.02 --lr 2e-3 --warmup-epochs 5 --aa=rand-m7-mstd0.5-inc1 --reprob 0.0 --remod 'pixel' --mixup 0.1 --cutmix 1.0 --seed 42 --min-lr 1.0e-06 --smoothing 0.0 --crop-pct 0.95 --val-split val --output OUTPUT_PATH --aug-repeats 3 --log-wandb --seed 1 --num-classes 1000 --model nexception-t
```

For NEXcepTion-TP run:
```
python -m torch.distributed.launch --nproc_per_node=4 train.py DATA_PATH --amp --batch-size 64 --opt fusedlamb --weight-decay 0.02 --sched cosine --input-size 3 224 224 --drop-path 0.05 --channels-last -j 8 --bce-loss --bce-target-thresh 0.2 --epochs 300 --weight-decay 0.02 --lr 2e-3 --warmup-epochs 5 --aa=rand-m7-mstd0.5-inc1 --reprob 0.0 --remod pixel --mixup 0.1 --cutmix 1.0 --seed 42 --min-lr 1.0e-06 --smoothing 0.0 --crop-pct 0.95 --val-split val --output OUTPUT_PATH --aug-repeats 3 --log-wandb --seed 1 --num-classes 1000 --model nexception-tp
```
For NEXcepTion-S run:
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --local_rank=0 DATA_PATH --amp --batch-size 32 --opt fusedlamb --weight-decay 0.02 --sched cosine --input-size 3 224 224 --drop-path 0.05 --channels-last -j 8 --bce-loss --bce-target-thresh 0.2 --epochs 300 --weight-decay 0.02 --lr 1.4e-3 --warmup-epochs 5 --aa=rand-m7-mstd0.5-inc1 --reprob 0.0 --remod pixel --mixup 0.1 --cutmix 1.0 --seed 42 --min-lr 1.0e-06 --smoothing 0.0 --crop-pct 0.95 --val-split val --output OUTPUT_PATH --aug-repeats 3 --log-wandb --seed 2 --num-classes 1000 --model nexception-s
```
