import os
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from data.dataset import get_train_dataset
from model.RMFG_Network import RMFGNet 


class PretrainSolver(object):
    def __init__(self, config=None, logging=None):
        self.model, self.optimizer, self.writer, self.criterion = None, None, None, None
        self.config = config
        self.logging = logging

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def bulid(self):
        self.model = RMFGNet().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.pretrain_lr, weight_decay=self.config.weight_decay)
        self.criterion = nn.BCELoss().cuda()
        self.tb = SummaryWriter(log_dir=self.config.train_tb_path)
        self.train_loader = data.DataLoader(dataset=get_train_dataset(),
                                   batch_size=self.config.train_batchsize,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=False)
        
    def adjust_learning_rate(self, optimizer, epoch, start_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = start_lr * (self.config.decay_rate ** (epoch // self.config.decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def clip_gradient(self, grad_clip):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    
    def train(self):
        images, labels, _ = next(iter(self.train_loader))
        self.tb.add_image('images', torchvision.utils.make_grid(images))
        self.tb.add_image('labels', torchvision.utils.make_grid(labels))
        self.tb.add_graph(self.model, images.cuda())
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=6e-9)
        for epoch in range(self.config.train_epoch):
            self.model.train()
            # self.adjust_learning_rate(self.optimizer, epoch, self.config.pretrain_lr)
            self.logging.info("Epoch:{}  Lr:{:.2E}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            print("Epoch:{}  Lr:{:.2E}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            
            loss_all, epoch_step = 0, 0
            for i, (images, gts, img_path) in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()
                images, gts = images.cuda(), gts.cuda()
                preds = self.model(images)
                preds = preds.squeeze().contiguous()

                loss = self.criterion(preds.squeeze(), gts.contiguous().view(-1, *(gts.shape[2:])).squeeze().float())
                # clip_gradient(optimizer, config.clip)
                loss_all += loss.data
                epoch_step += 1
                loss.backward()
                self.optimizer.step()

                if i % 20 == 0 or i == len(self.train_loader) or i == 1:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                        format(datetime.now(), epoch + 1, self.config.train_epoch, i, len(self.train_loader), loss.data))
                    self.logging.info(
                        '[Pretrain]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                        format(epoch + 1, self.config.train_epoch, i, len(self.train_loader), loss.data))

            loss_all /= epoch_step
            self.logging.info('[Pretrain]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch + 1, self.config.train_epoch, loss_all))
            print('[Pretrain]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch + 1, self.config.train_epoch, loss_all))
            self.tb.add_scalar('Pretrain Loss', loss_all, epoch)

            scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.config.train_save_path, "Pretrain_%d.pth" % (epoch + 1)))
    
    def close(self):
        self.tb.close()