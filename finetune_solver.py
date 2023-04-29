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

from data.dataset import get_finetune_dataset
from model.RMFG_Network import RMFGNet 


class FinetuneSolver(object):
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
        if self.config.train_state_dict is not None:
            self.model.load_backbone(torch.load(self.config.train_state_dict, map_location=torch.device('cpu')), self.logging)
            print('load model from ', self.config.train_state_dict)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.finetune_lr, weight_decay=self.config.weight_decay)
        self.criterion = nn.BCELoss().cuda()
        self.tb = SummaryWriter(log_dir=self.config.finetune_tb_path)
        self.finetune_loader = data.DataLoader(dataset=get_finetune_dataset(),
                                   batch_size=self.config.finetune_batchsize,
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
        images, labels = next(iter(self.finetune_loader))
        self.tb.add_image('images', torchvision.utils.make_grid(images.view(-1, *(images.shape[2:]))))
        self.tb.add_image('labels', torchvision.utils.make_grid(labels.view(-1, *(labels.shape[2:]))))
        self.tb.add_graph(self.model, images.cuda())
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=6e-9)
        for epoch in range(self.config.finetune_epoch):
            self.model.train()
            # self.adjust_learning_rate(self.optimizer, epoch, self.config.finetune_lr)
            self.logging.info("Epoch:{}  Lr:{:.2E}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            print("Epoch:{}  Lr:{:.2E}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            
            loss_all, epoch_step = 0, 0
            for i, (images, gts) in enumerate(self.finetune_loader, start=1):
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

                if i % 20 == 0 or i == len(self.finetune_loader) or i == 1:
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                        format(datetime.now(), epoch + 1, self.config.finetune_epoch, i, len(self.finetune_loader), loss.data))
                    self.logging.info(
                        '[Finetune]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                        format(epoch + 1, self.config.finetune_epoch, i, len(self.finetune_loader), loss.data))

            loss_all /= epoch_step
            self.logging.info('[Finetune]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch + 1, self.config.finetune_epoch, loss_all))
            print('[Finetune]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch + 1, self.config.finetune_epoch, loss_all))
            self.tb.add_scalar('Finetune Loss', loss_all, epoch)

            scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.config.finetune_save_path, "Finetune_%d.pth" % (epoch + 1)))
    
    def close(self):
        self.tb.close()