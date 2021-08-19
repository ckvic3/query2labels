import json
import numpy as np
import os
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from lib.models.backbone import build_backbone
from lib.models.transformer import build_transformer
from lib.utils.misc import clean_state_dict
from lib.utils.logger import setup_logger
import lib.models as models
import lib.models.aslloss
import numpy as np
import argparse
from lib.models.query2label import build_q2l
from lib.utils.metric import voc_mAP
from lib.utils.slconfig import get_raw_dict
from q2l_infer import AverageMeter
import random

from lib.dataset.get_dataset import get_datasets
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
# 单机多卡版本
# local rank 代表的是rank 版本号
def parser_args():
    available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384', 'Q2L-CvT_w24-384']

    parser = argparse.ArgumentParser(description='Query2Label for multilabel classification')
    parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14'])
    parser.add_argument('--root', help='data root path', default='/data/coco')


    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +
                            ' | '.join(available_models) +
                            ' (default: Q2L-R101-448)')
    parser.add_argument('--config', default=None, type=str, help='config file')

    parser.add_argument('--output', metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--loss', metavar='LOSS', default='asl',
                        choices=['asl'],
                        help='loss functin')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of classes.")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=80, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--learning', action='store_true')

    # optim method
    parser.add_argument('--lr',default=1e-4)

    # loss function
    parser.add_argument('--gamma_neg', default=4)
    parser.add_argument('--gamma-pos', default=0)
    parser.add_argument('--eps', default=1e-5, type=float,
                    help='eps for focal loss (default: 1e-5)')

    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help='use mixture precision.')
    # data aug
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    args = parser.parse_args()

    # update parameters with pre-defined config file
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k,v in cfg_dict.items():
            setattr(args, k, v)
    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    torch.cuda.set_device(args.local_rank)

    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.local_rank)
    # 一种优化策略， 适合模型训练过程中输入大小不发生改变 ：https://zhuanlan.zhihu.com/p/73711222
    cudnn.benchmark = True

    # set output dir and logger

    if not args.output:
        now = datetime.datetime.now()
        args.output = (f"logs/{args.arch}-{now.date()}-{now.hour}-{now.minute}").replace(' ', '-')
    os.makedirs(args.output, exist_ok=True)
    dist.barrier()
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: " + ' '.join(sys.argv))

    # save config to outputdir
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('local_rank: {}'.format(args.local_rank))

    global writer
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=args.output)

    # build model
    model = build_q2l(args)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=True,
        eps=args.eps,
    )

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)

    # 让一个batch的图片可以均匀分布到多卡上
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    # Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    optimizer = torch.optim.AdamW(params= model.parameters(), lr=args.lr, betas=(0.9, 0.9999))

    steps_per_epoch = len(train_loader) // dist.get_world_size()
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs,
                                        pct_start=0.2)

    best_mAp_score = 0.0

    for epoch in range(args.epochs):
        # train one epoch
        train(train_loader, model, criterion, optimizer, scheduler, args, logger, epoch, steps_per_epoch)
        _, mAp_score = validate(val_loader, model, criterion, args, logger)

        if dist.get_rank() == 0:
            writer.add_scalar('train/map', mAp_score, epoch)
            if mAp_score > best_mAp_score:
                best_mAp_score = mAp_score
                # save checkpoint
                save_checkpoint(model.state_dict(),True)
            print("current mAp score : {}, best mAp score : {} ".format(mAp_score,best_mAp_score))


def train(train_loader, model, criterion, optimizer, scheduler, args, logger, epoch ,steps_per_epoch):
    model.train()
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    saved_data = []

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, mem],
        prefix='Train: ')

    scaler = GradScaler()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast():
            output = model(images)
            if len(output) > 1:
                loss = criterion(output[0], target) + criterion(output[1], target)
            else:
                loss = criterion(output[-1], target)
            output_sm = nn.functional.sigmoid(output[-1])

        # loss backward
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # lr update
        scheduler.step()

        # record loss
        losses.update(loss.item(), images.size(0))
        # record mem use
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        #
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss', losses.avg, epoch * steps_per_epoch + i)

        # save some data
        _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
        saved_data.append(_item)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and dist.get_rank() == 0:
            progress.display(i, logger)

    logger.info('=> synchronize...')
    if dist.get_world_size() > 1:
        dist.barrier()
    loss_avg, = map(
        _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
        [losses]
    )

    # calculate mAP
    saved_data = torch.cat(saved_data, 0).numpy()
    saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name), saved_data)
    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0:  # 只在master进程进行计算
        print("Calculating mAP:")
        filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        metric_func = voc_mAP
        mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class,
                               return_each=True)

        logger.info("  mAP: {}".format(mAP))
        logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
    else:
        mAP = 0

    if dist.get_world_size() > 1:
        dist.barrier()

    return loss_avg, mAP


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                # loss = criterion(output, target)
                output_sm = nn.functional.sigmoid(output)

            # record loss
            # losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        # loss_avg, = map(
        #     _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
        #     [losses]
        # )

        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP
            mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class,
                                   return_each=True)

            logger.info("  mAP: {}".format(mAP))
            logger.info("   aps: {}".format(np.array2string(aps, precision=5)))
        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return None, mAP

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, './checkpoint' + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

if __name__ == '__main__':
    main()