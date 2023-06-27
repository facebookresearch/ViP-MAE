# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import setup_seed, set_grad_to_vec

import models_mae
from engine_pretrain import train_one_epoch
import torch.distributed as dist
import math
import opacus
from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from opacus.data_loader import DPDataLoader
import functorch
from tqdm import tqdm
import socket
import submitit
from lion_pytorch import Lion
from opacus.data_loader import DPDataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.optimizers.optimizer import DPOptimizer


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Overall batch size')
    parser.add_argument('--max_device_batch_size', type=int, default=12)
    parser.add_argument('--epochs', default=10000, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_dec512d4b', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay (default: 0.005)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N',
                        help='warm up steps')
    parser.add_argument('--total_steps', type=int, default=10000, metavar='N',
                        help='total steps')
    parser.add_argument('--lr_decay', default=1, type=int)
    parser.add_argument('--beta2', type=float, default=0.95)

    # DP parameters
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=1e-9)
    parser.add_argument('--max_norm', type=float, default=0.1)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417', type=str,
                        help='dataset path')
    parser.add_argument('--data_pretrain', default='imagenet1k', type=str,
                        help='dataset name')
    parser.add_argument('--data_subsample_ratio', default=1.0, type=float,
                        help='dataset subsampling')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # others
    parser.add_argument('--print_freq', default=50, type=int,
                        help='print frequency')
    parser.add_argument('--save_freq', default=1000, type=int,
                        help='save ckpt frequency')
    return parser


def load_imgs(udata):
    # -- unsupervised imgs
    imgs = [u.to(device, non_blocking=True) for u in udata]
    return imgs


def main(args):
    misc.init_distributed_mode(args)
    world_size = misc.get_world_size()

    if world_size > 1:
        # print master address
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        
        hostname = socket.gethostname()
        print('hostname: ', hostname)
        local_ip_address = socket.gethostbyname(hostname)
        print('local_ip_address: ', local_ip_address)    
        print('args.gpu: ', args.gpu)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('device: ', device)
    print('world_size: ', world_size)

    # all nodes MUST use the same seed to generate the same noise!
    setup_seed(args.seed)
    cudnn.benchmark = True

    # setup batch size
    batch_size = args.batch_size

    # construct/load dataset
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    print('finished constructing dataset')
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # construct data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    # wrap data loader with DPDataLoader
    distributed_loader = world_size > 1
    data_loader_train = DPDataLoader.from_data_loader(data_loader_train, 
                                                      generator=None, 
                                                      distributed=distributed_loader)
    print('wrap with DP loader')

    print('=============================================')
    print('loaded ImageNet-1K dataset')
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    # load synthetic pre-trained model
    misc.load_synthetic_pretrain_model(args=args, model_without_ddp=model)

    # dist setup
    if world_size > 1:
        for w in model.parameters():
            dist.broadcast(w, src=0)
    
    # model_without_ddp = model
    print("Model = %s" % str(model))

    # setup learning rate    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256

    print("base lr: %.2e" % args.blr)
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % (args.batch_size))

    # Train all the parameters
    for param in model.parameters():
        param.requires_grad = True
    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay)

    # construct dummy model and dummy optimizer
    theta_dummy = torch.tensor([0.0], requires_grad=True)
    optimizer_dummy = torch.optim.AdamW([theta_dummy], lr=0.0)
    dp_optimizer_dummy = DPOptimizer(optimizer=optimizer_dummy,
                                     noise_multiplier=1.0, 
                                     max_grad_norm=1.0,
                                     expected_batch_size=args.max_device_batch_size)

    # lr scheduler
    if args.lr_decay:
        lr_func = lambda step: min((step + 1) / (args.warmup_steps + 1e-8), 0.5 * (math.cos(step / args.total_steps * math.pi) + 1))
    else:
        lr_func = lambda step: min((step + 1) / (args.warmup_steps + 1e-8), 1.0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)
    lr_current = optimizer.param_groups[0]["lr"]
    print('start with learning rate: ', lr_current)

    # count number of parameters
    if global_rank == 0:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('number of params: ', params)
        print('============================================================')
    
    loss_scaler = NativeScaler()

    ### DP setup
    start_epoch = 0
    step_count = 0

    # wrap model/optimizer/loader with opacus
    q = float(args.batch_size) / len(dataset_train)
    print('sample rate (q): ', q)
    args.delta = 1.0 / (2 * len(dataset_train))
    orders = np.array(list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)))
    rdp_const = compute_rdp(q=q, noise_multiplier=args.sigma, steps=args.total_steps, orders=orders)
    epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=args.delta)
    print(f"Estimated total epsilon={epsilon:.2f} at delta={args.delta}")
    print('total_steps: ', args.total_steps)

    ### functorch utils
    func_model, weights = functorch.make_functional(model)
    def compute_loss(weights, X):
        predicted_X, mask, target_img = func_model(weights, X.unsqueeze(0), args.mask_ratio)
        loss = (((predicted_X - target_img) ** 2).mean(dim=-1) * mask).sum() / mask.sum()
        return loss
    compute_grad_and_loss = lambda weights, X: functorch.vmap(functorch.grad_and_value(compute_loss), (None, 0), randomness='different')(weights, X)

    print(f"Start training for {args.total_steps} steps")
    start_time = time.time()

    num_samples_accum = torch.zeros(1).to(device)
    losses_sum_accum = torch.zeros(1).to(device)
    
    # DP training
    optimizer.zero_grad()
    for e in range(start_epoch, args.epochs):

        model.train()

        grad_vec = None
        _, weights = functorch.make_functional(model)
        num_samples = torch.zeros(1).to(device)
        losses_sum = torch.zeros(1).to(device)

        with BatchMemoryManager(
            data_loader=data_loader_train, 
            max_physical_batch_size=args.max_device_batch_size, 
            optimizer=dp_optimizer_dummy
        ) as memory_safe_data_loader:

            for img, _ in tqdm(iter(memory_safe_data_loader)):
                # check whether skip
                if dp_optimizer_dummy._step_skip_queue:
                    do_step_skip = dp_optimizer_dummy._step_skip_queue.pop(0)
                else:
                    do_step_skip = False

                # compute (per-sample) gradient
                img = img.to(device)
                model.zero_grad()
                grads, loss = compute_grad_and_loss(weights, img)

                num_samples += len(loss)
                losses_sum += loss.sum().detach().item()

                with torch.no_grad():
                    # flatten gradient
                    grad_tensor = []
                    for grad in grads:
                        grad_tensor.append(grad.view(grad.size(0), -1))
                    del grads
                    grad_tensor = torch.cat(grad_tensor, 1)

                    # gradient clipping
                    grad_norm = grad_tensor.norm(2, 1)
                    multiplier = grad_norm.new(grad_norm.size()).fill_(1)
                    multiplier[grad_norm.gt(args.max_norm)] = args.max_norm / grad_norm[grad_norm.gt(args.max_norm)]
                    grad_tensor *= multiplier.unsqueeze(1)

                    # accumulate clipped (per-sample) gradient
                    if grad_vec is None:
                        grad_vec = grad_tensor.sum(0)
                    else:
                        grad_vec += grad_tensor.sum(0)

                # aggregate gradient and perform one step of update
                if not do_step_skip:
                    # synchronization
                    if world_size > 1:
                        dist.barrier()

                    # step count for rdp accounting
                    step_count += 1 
                    
                    # generate noise vector, ensure every machine has the same noise_vec
                    noise_vec = torch.randn_like(grad_vec).to(device) * 0.0
                    # only generate the noise_vec on the master machine, then broadcast to other machines
                    if global_rank == 0:
                        noise_vec = torch.randn_like(grad_vec).to(device)
                    
                    # reduce gradient sum across nodes
                    if world_size > 1:
                        dist.barrier()
                        # sum up grad_vec over all machines
                        dist.all_reduce(grad_vec, op=dist.ReduceOp.SUM)
                        # broadcast noise_vec to all machines
                        dist.all_reduce(noise_vec, op=dist.ReduceOp.SUM)
                        # sum up loss and num of samples
                        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
                        dist.all_reduce(losses_sum, op=dist.ReduceOp.SUM)

                    ### ADD NOISE ###
                    grad_vec = (grad_vec + args.sigma * args.max_norm * noise_vec) / num_samples

                    # optimize step
                    set_grad_to_vec(model, grad_vec)
                    optimizer.step()

                    # reset grad_vec
                    grad_vec = None

                    # reset the weights for functorch
                    _, weights = functorch.make_functional(model)

                    # lr schedule
                    lr_scheduler.step()
                
                    # compute (avg) loss
                    losses_sum_accum += losses_sum
                    num_samples_accum += num_samples
                    loss_avg_iter_ = losses_sum_accum * 1.0 / num_samples_accum

                    # reset num of samples and sum of losses
                    num_samples = torch.zeros(1).to(device)
                    losses_sum = torch.zeros(1).to(device)

                    # compute rdp and eps
                    rdp_const = compute_rdp(q=q, noise_multiplier=args.sigma, steps=step_count, orders=orders)
                    epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp_const, delta=args.delta)

                    # display current lr/loss/step
                    lr_current = optimizer.param_groups[0]["lr"]

                    if global_rank == 0 and (step_count % args.print_freq == 0):
                        print("Epoch [{}], Steps [{}], loss={:.6f} (eps={:.6f}, lr={:.6f}, (accum) num of samples={})".format(
                              e, step_count, loss_avg_iter_.item(), epsilon, lr_current, num_samples_accum))
                    
                    if step_count % args.print_freq == 0:
                        # reset number of samples and sum of loss
                        num_samples_accum = torch.zeros(1).to(device)
                        losses_sum_accum = torch.zeros(1).to(device)
                    
                    # synchronization 
                    if world_size > 1:
                        dist.barrier()

                    if args.output_dir and (step_count % args.save_freq == 0 or step_count == 1):
                        misc.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                                        loss_scaler=loss_scaler, epoch=step_count)
                    
                    # terminate if finish total_steps 
                    if step_count > args.total_steps:
                        break

        print("Epoch [{}] (finished), loss={:.6f} (eps={:.6f})".format(e, loss_avg_iter_.item(), epsilon))
        
        # terminate if finish total_steps 
        if step_count > args.total_steps:
            break

    print("(finished training), loss={:.6f} (eps={:.6f})".format(loss_avg_iter_.item(), epsilon))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
