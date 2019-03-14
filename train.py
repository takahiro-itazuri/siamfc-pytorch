from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC

from options import TrainOptions


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
    tracker = TrackerSiamFC(name=opt.name)

    # training loop
    for epoch in range(opt.num_epochs):
        for step, batch in enumerate(loader):
            loss = tracker.step(
                batch, backward=True, update_lr=(step == 0))
            if step % opt.print_freq == 0:
                print('Epoch [{}/{}][{}/{}]: Loss: {:.5f}'.format(
                    epoch + 1, opt.num_epochs, step + 1, len(loader), loss))
                sys.stdout.flush()

        # save checkpoint
        net_path = os.path.join(opt.log_dir, 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
