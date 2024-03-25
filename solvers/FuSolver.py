import os
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import scipy
import spectral as spy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil


# from models.vgg import vgg16
from networks import create_model

from .base_solver import BaseSolver
from networks import init_weights
import torch.nn.functional as F
from utils import util



class FuSolver(BaseSolver):
    def __init__(self, opt):
        super(FuSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.Vi = self.Tensor()

        # self.HR = self.Tensor()
        self.Ir = self.Tensor()

        self.fused_img_cr =  self.Tensor()
        self.fused_img_cb =  self.Tensor()
        # self.PAN_unalign = self.Tensor()
        self.lossWeight = None
        self.Fu = None

        self.records = {'train_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'lr': []}

        self.model = create_model(opt)
        self.print_network()

        if self.is_train:
            self.model.train()
            # set loss
            self._set_loss()
            # set optimizer
            self._set_optimizer()
            # set lr_scheduler
            self._set_scheduler()

        self._load()

        print(f'===> Solver Initialized : [{self.__class__.__name__}] ||  Use GPU : [{self.use_gpu}]')

    def feed_data(self, batch):
        input = batch['Vi']
        input_ = batch['Ir']
        # a = input.shape[2] % 4
        # b = input.shape[3] % 4
        # left = nn.ReplicationPad2d((1, 0, 0, 0))
        # upper = nn.ReplicationPad2d((0, 0, 1, 0))
        # double = nn.ReplicationPad2d((0, 1, 1, 0))
        # while a % 4 != 0 and b % 4 != 0:
        #     input_ = double(input_)
        #     input = double(input)
        #     a += 1
        #     b += 1
        # while a % 4 != 0:
        #     input_ = upper(input_)
        #     input = upper(input)
        #     a += 1
        # while b % 4 != 0:
        #     input_ = left(input_)
        #     input = left(input)
        #     b += 1
        # input_pan_unalign = batch["pan_unalign"]
        self.Vi.resize_(input.size()).copy_(input)  # 将数据复制到对应的设备上
        self.Ir.resize_(input_.size()).copy_(input_)
        # self.fused_img_cr = batch['fused_img_cr']
        # self.fused_img_cb = batch['fused_img_cb']
        # self.PAN_unalign.resize_(input_pan_unalign.size()).copy_(input_pan_unalign)

    def train_step(self):
        # self.weight_list = self.generatLossWeight()

        self.model.train()
        self.optimizer.zero_grad()

        loss_batch = 0.0
        sub_batch_size = int(self.Vi.size(0) / self.split_batch)
        for i in range(self.split_batch):
            loss_sbatch = 0.0
            split_Vi = self.Vi.narrow(0, i * sub_batch_size, sub_batch_size)   #切片。和没切没变化
            # split_HR = self.HR.narrow(0, i * sub_batch_size, sub_batch_size)
            split_Ir = self.Ir.narrow(0, i * sub_batch_size, sub_batch_size)
            # split_PAN_unalign = self.PAN_unalign.narrow(0, i * sub_batch_size, sub_batch_size)

            q = self.train_opt['q']
            if q == 'vi':
                output = self.model(split_Vi, split_Ir)
            elif q == 'ir':
                output = self.model(split_Ir,split_Vi)
            else:
                output = self.model(split_Ir, split_Vi)

            loss_sbatch = self.criterion_pix(output, split_Vi, split_Ir)      #output, split_HR, split_LR, split_PAN

            loss_sbatch /= self.split_batch
            loss_sbatch.backward()

            loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
            print(f'[Warning] Skip this batch! (Loss: {loss_batch})')

        self.model.eval()
        return loss_batch

    # def generatLossWeight(self):
    #     self.feature_model = vgg16().cuda()
    #     self.feature_model.load_state_dict(torch.load('./models/vgg16.pth'))
    #     c = 3500
    #     vi_a = (self.Vi + 1) / 2
    #     vi_a = vi_a.cuda()
    #     Ir_a = (self.Ir + 1) / 2
    #     Ir_a = Ir_a.cuda()
    #     with torch.no_grad():
    #         feat_1 = torch.cat((vi_a, vi_a, vi_a), dim=1)
    #         feat_1 = self.feature_model(feat_1)
    #         feat_2 = torch.cat((Ir_a, Ir_a, Ir_a), dim=1)
    #         feat_2 = self.feature_model(feat_2)
    #
    #         for i in range(len(feat_1)):
    #             m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
    #             m2 = torch.mean(features_grad(feat_2[i]).pow(2), dim=[1, 2, 3])
    #             if i == 0:
    #                 w1 = torch.unsqueeze(m1, dim=-1)
    #                 w2 = torch.unsqueeze(m2, dim=-1)
    #             else:
    #                 w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
    #                 w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
    #         weight_1 = torch.mean(w1, dim=-1) / c
    #         weight_2 = torch.mean(w2, dim=-1) / c
    #         weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
    #         weight_list = F.softmax(weight_list, dim=-1)
    #     return weight_list
    def get_LossWeight(self,output):
        min_v = np.min(output["weight"])
        max_v = np.max(output["weight"])
        img_v = (output["weight"] - min_v) / (max_v - min_v)
        return img_v

    def test(self):
        self.model.eval()
        with torch.no_grad():
            q = self.train_opt['q']
            if q == 'vi':
                self.Fu = self.model(self.Vi, self.Ir)
            elif q == 'ir':
                self.Fu = self.model(self.Ir, self.Vi)
            else:
                self.Fu = self.model(self.Ir, self.Vi)

        self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.Fu, self.Vi, self.Ir)
            return loss_pix.item()

    def visualization(self):
        return

    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]' % filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp', 'best_ckp'))
            torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp',
                                                                                 'epoch_%d_ckp.pth' % epoch)))

            torch.save(ckp, filename.replace('last_ckp', 'epoch_%d_ckp.pth' % epoch))

    def _load(self):
        """
        load or initialize network
        """

        if self.is_train and not self.opt['solver']['pretrain']:  # 不是续训练
            self._net_init()
        elif self.is_train and self.opt['solver']['pretrain'] == 'resume':  # 是续训练
            model_path = self.opt['solver']['pretrained_path']
            if not os.path.exists(model_path):
                raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.cur_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.best_epoch = checkpoint['best_epoch']
            self.records = checkpoint['records']
        else:
            model_path = self.opt['solver']['pretrained_path']
            if not os.path.exists(model_path): raise ValueError(
                "[Error] The 'pretrained_path' does not declarate in *.json")
            checkpoint = torch.load(model_path)
            if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
            load_func = self.model.load_state_dict if isinstance(self.model, nn.DataParallel) \
                else self.model.module.load_state_dict
            load_func(checkpoint)

    def get_K_V(self):
        kv_dict = OrderedDict()
        kv_dict['k'] = self.Fu["k"].data[0].float().cpu()
        kv_dict['v'] = self.Fu["v"].data[0].float().cpu()
        return kv_dict

    def get_pre(self):
        kv_dict = OrderedDict()
        kv_dict['outPre'] = self.Fu["outPre"].data[0].float().cpu()
        return kv_dict

    def save_K_V(self,visuals,save_kv_path,index):
        # visuals = self.get_K_V()
        length = len(visuals['k'].numpy())
        vis_k = visuals['k'].numpy().transpose(1, 2, 0)
        vis_v = visuals['v'].numpy().transpose(1, 2, 0)
        if not os.path.exists(save_kv_path + f"/K/{index}"): os.makedirs(save_kv_path + f"/K/{index}")
        if not os.path.exists(save_kv_path + f"/V/{index}"): os.makedirs(save_kv_path + f"/V/{index}")
        for i in range(length):
            k = vis_k[:,:,i]
            v = vis_v[:,:,i]
            k_path = save_kv_path + f"/K/{index}/{i}k.png"
            v_path = save_kv_path + f"/V/{index}/{i}v.png"
            cv2.imwrite(k_path, k)
            cv2.imwrite(v_path, v)

    def get_current_visual(self, need_np=False, need_HR=True):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        out_dict['Vi'] = self.Vi.data[0].float().cpu()
        out_dict['Ir'] = self.Ir.data[0].float().cpu()
        out_dict["Fu"] = self.Fu["pred"].data[0].float().cpu()
        # out_dict['fused_img_cr'] = self.fused_img_cr.data[0].float().cpu()
        # out_dict['fused_img_cb'] = self.fused_img_cb.data[0].float().cpu()

        # out_dict["weight"] = self.Fu["weight"].data[0].float().cpu()
        # out_dict['SR'] = self.SR.data[0].float().cpu()
        # out_dict['HR'] = self.HR.data[0].float().cpu()

        return out_dict

    def save_current_visual(self, epoch, img_num):
        """
        save visual results for comparison
        """
        # if epoch % self.save_vis_step == 0:
        visuals = self.get_current_visual(need_np=False)
        Vi = visuals['Vi'].numpy().transpose(1, 2, 0)
        Fu = visuals['Fu'].numpy().transpose(1, 2, 0)
        Ir = visuals['Ir'].numpy().transpose(1, 2, 0)
        # HR = visuals['HR'].numpy().transpose(1, 2, 0)


        # Vi = (Vi + 1) * 127.5
        # Ir = (Ir + 1) * 127.5
        # Fu = (Fu + 1) * 127.5

        if not os.path.exists(self.visual_dir + "/"+str(epoch)+"/Vi/"): os.makedirs(self.visual_dir + "/"+str(epoch)+"/Vi/")
        if not os.path.exists(self.visual_dir + "/"+str(epoch)+"/Fu/"): os.makedirs(self.visual_dir + "/"+str(epoch)+"/Fu/")
        if not os.path.exists(self.visual_dir + "/"+str(epoch)+"/Ir/"): os.makedirs(self.visual_dir + "/"+str(epoch)+"/Ir/")

        Vi_path = self.visual_dir + "/"+str(epoch)+"/Vi/"+str(img_num)+".png"
        # spy.save_rgb(Vi_path, Vi, bands=[2, 1, 0])
        # spy.save_rgb(Vi_path, Vi, bands=[0])
        cv2.imwrite(Vi_path, Vi)

        Fu_path = self.visual_dir + "/"+str(epoch)+"/Fu/"+str(img_num)+".png"
        # spy.save_rgb(Fu_path, Fu, bands=[0])
        cv2.imwrite(Fu_path, Fu)

        Ir_path = self.visual_dir + "/"+str(epoch)+"/Ir/"+str(img_num)+".png"
        # spy.save_rgb(Ir_path, Ir, bands=[0])
        cv2.imwrite(Ir_path, Ir)
        # HR_path = self.visual_dir + f"/{epoch}/{img_num}/HR.bmp"
        # spy.save_rgb(HR_path, HR, bands=[2, 1, 0])

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']

    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss'][-1]
                , 'val_loss': self.records['val_loss'][-1]
                , 'psnr': self.records['psnr'][-1].item()
                , 'ssim': self.records['ssim'][-1].item()
                , 'lr': self.records['lr'][-1]
                  },
            index=range(self.cur_epoch, self.cur_epoch + 1)
        )
        is_need_header = True if self.cur_epoch == 1 else False
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'), mode="a",
                          index_label='epoch', header=is_need_header)

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                             self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")

    def _set_loss(self):
        loss_type = self.train_opt['loss_type']
        if loss_type == 'l1':
            self.criterion_pix = nn.L1Loss()
        elif loss_type == "loss":
            from networks.loss import fusion_loss_med
            self.criterion_pix = fusion_loss_med()

        else:
            raise NotImplementedError('Loss type [%s] is not implemented!' % loss_type)

        if self.use_gpu:
            self.criterion_pix = self.criterion_pix.cuda()

    def _set_optimizer(self):
        weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
        optim_type = self.train_opt['type'].upper()
        if optim_type == "ADAM":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
        elif optim_type == "ADAMW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_opt['learning_rate'],
                                         weight_decay=weight_decay)
        else:
            raise NotImplementedError('Optimizer type [%s] is not implemented!' % optim_type)

    def _set_scheduler(self):
        if self.train_opt['lr_scheme'].lower() == 'multisteplr':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            self.train_opt['lr_steps'],
                                                            self.train_opt['lr_gamma'])

        else:
            raise NotImplementedError('Only MultiStepLR scheme is supported!')
        print("optimizer: ", self.optimizer)
        print(f"lr_scheduler milestones: {self.scheduler.milestones}   gamma: {self.scheduler.gamma:.3f}")

    def _net_init(self, init_type='normal'):
        print('==> Initializing the network using [%s]' % init_type)
        init_weights(self.model, init_type)
