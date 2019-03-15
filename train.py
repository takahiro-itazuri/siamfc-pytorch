from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC

from options import TrainOptions
from meter import AverageMeter
from logger import Logger


if __name__ == '__main__':
    opt = TrainOptions().parse()

    # setup dataset
    if opt.dataset in ['GOT-10k', 'GOT10k', 'got10k', 'got-10k']:
        root_dir = 'data/GOT-10k'
        seq_dataset = GOT10k(root_dir, subset='train')
    elif opt.dataset in ['ILSVRC', 'ilsvrc']:
        root_dir = 'data/ILSVRC'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    else:
        raise NotImplementedError
    pair_dataset = Pairwise(seq_dataset)

    # setup data loader
    loader = DataLoader(
        pair_dataset, batch_size=opt.batch_size, shuffle=True,
        pin_memory=opt.cuda, drop_last=True, num_workers=opt.num_workers)

    # setup tracker
    tracker = TrackerSiamFC(name=opt.name, weight=opt.weight, device=opt.device)

    # training loop
    itr = 0
    num_itrs = int((opt.num_epochs * len(loader)) / opt.print_freq) + 1
    loss_logger = Logger(os.path.join(opt.log_dir, 'loss.csv'), num_itrs)
    loss_meter = AverageMeter()
    for epoch in range(opt.num_epochs):
        for step, batch in enumerate(loader):
            loss = tracker.step(
                batch, backward=True, update_lr=(step == 0))

            itr += 1
            loss_meter.update(loss)
            if itr % opt.print_freq == 0:
                print('Epoch [{}/{}] itr [{}]: Loss: {:.5f}'.format(
                    epoch + 1, opt.num_epochs, itr, loss_meter.avg))
                sys.stdout.flush()

                loss_logger.set(itr / opt.print_freq, loss_meter.avg)
                loss_meter = AverageMeter()

        # save checkpoint
        net_path = os.path.join(opt.log_dir, 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
