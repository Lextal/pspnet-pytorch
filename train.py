import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pspnet import PSPNet

import logging
import click
import os
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def weights_log(class_freq):
    weights = torch.log1p(1 / class_freq)
    return weights / torch.sum(weights)


def lr_poly(base_lr, epoch, max_epoch, power):
    return max(0.00001, base_lr * np.power(1. - epoch / max_epoch, power))


def build_network(snapshot):
    epoch = 0
    net = PSPNet()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


@click.command()
@click.option('--data-path', type=str, help='Path to dataset with directories imgs/ maps/')
@click.option('--models-path', type=str, help='Path for storing model snapshots')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=200)
@click.option('--crop_y', type=int, default=300)
@click.option('--batch-size', type=int, default=1)
@click.option('--alpha', type=float, default=5.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0')
@click.option('--start-lr', type=float, default=0.01)
@click.option('--lr-power', type=float, default=0.9)
def train(data_path, models_path, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, lr_power, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot)
    steps = 0

    for epoch in range(starting_epoch, starting_epoch + epochs):

        # You have to load all this stuff by yourself
        # class_weights is simply a 1d normalized Tensor
        # n_images is used to calculate the "poly" LR

        loader, class_weights, n_images = None, None, None

        n_images *= epochs
        seg_criterion = nn.NLLLoss2d(weight=class_weights.cuda())
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        epoch_losses = []
        for x, y, y_cls in loader:
            steps += batch_size
            lr = lr_poly(start_lr, steps, n_images, lr_power)
            optimizer = optim.Adam(net.parameters(), lr=lr)
            optimizer.zero_grad()
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            y_cls = Variable(y_cls).cuda()
            out, out_cls = net(x)
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha * cls_loss
            logging.info(
                'Step {4}/{5} : Seg loss = {0:0.5f}, Cls loss = {1:0.5f}, Total = {2:0.5f}, LR = {3:0.5f}'.format(
                    seg_loss.data[0], cls_loss.data[0], loss.data[0], lr, steps, n_images))
            loss.backward()
            optimizer.step()
        logging.info('Epoch = {0}, Loss = {1:0.5f}'.format(epoch, np.mean(epoch_losses)))
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch)])))


if __name__ == '__main__':
    train()