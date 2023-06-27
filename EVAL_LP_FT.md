## Linear Probing and Fine-tuning DP Pre-trained ViP for Classification

### Evaluation

As a sanity check, run evaluation using our private pre-trained **ViP** models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViP-Syn-Base</th>
<th valign="bottom">ViP-Base</th>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet accuracy (linear probing)</td>
<td align="center">49.8</td>
<td align="center">55.7</td>
</tr>
</tbody></table>

#### Quick start (CIFAR10 linear probing)
Evaluate (linear probing) **ViP-Base** in a single GPU on CIFAR10:
```
python main_linprobe.py \
    --batch_size 128 \
    --global_pool \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --data_path ${CIFAR10_DIR} \
    --nb_classes 19 \
    --dataset_name "CIFAR10"
```

## Multi-node Distributed Training
- Install submitit (`pip install submitit`) first.


### Linear Probing (ImageNet)
To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each:
```
python submitit_linprobe.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --ngpus 8 \
    --batch_size 512 \
    --global_pool \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --partition ${PARTITION_NAME} \
    --dist_eval \
    --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 512 (`batch_size` per gpu) * 4 (`nodes`) * 8 (gpus per node) = 16384.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- To run single-node training, follow the instruction in the *Quick start (CIFAR10 linear probing)* part.

### Fine-tuning

To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each:
```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --ngpus 8 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --partition ${PARTITION_NAME} \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`nodes`) * 8 (gpus per node) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.

- Specify ``PRETRAIN_CHKPT`` variable with the path to the synthetic pre-trained ViP/ViP-Syn model.
- Specify ``IMAGENET_DIR`` (or ``CIFAR10_DIR``) variable with the path to the ImageNet (or CIFAR10) dataset.
- Specify ``JOB_DIR`` variable to define the directory for saving logs and checkpoints.
- Specify ``PARTITION_NAME`` variable with the slurm partition name.
