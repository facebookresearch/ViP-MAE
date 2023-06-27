## Pre-training ViP

To pre-train ViP-Base (recommended default) with **multi-node distributed training** on **ImageNet-1K**, run the following on 8 nodes with 8 GPUs each:
```
python submitit_pretrain_vip.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --ngpus 8 \
    --use_volta32 \
    --batch_size 81920 \
    --sigma 0.5 \
    --total_steps 10000 \
    --blr 1.5e-05 \
    --mask_ratio 0.75 \
    --weight_decay 0.005 \
    --max_device_batch_size 12 \
    --partition ${PARTITION_NAME} \
    --model mae_vit_base_dec512d4b \
    --max_norm 0.1 \
    --print_freq 10 \
    --data_path ${IMAGENET_DIR} \
    --resume ${CKPT_SynViP}
```
- Here the batch size is 81920. If memory or # gpus is limited, use `--max_device_batch_size` to reduce the batch size in each (accumulation) gradient step on each gpu.
- Specify ``CKPT_SynViP`` variable with the path to the synthetic pre-trained ViP model.
- Specify ``IMAGENET_DIR`` variable with the path to the ImageNet dataset.
- Specify ``JOB_DIR`` variable to define the directory for saving logs and checkpoints.
- Specify ``PARTITION_NAME`` variable with the slurm partition name.
