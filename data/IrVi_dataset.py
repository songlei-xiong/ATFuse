import random

import torch
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
import numpy as np
from data import common


class IrViDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    可见光代替LR,红外代替Pan
    '''

    def name(self):
        return self.dataset_name  # 返回使用的数据集名称

    def __init__(self, opt):
        super(IrViDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']  # 放大倍数
        self.paths_Vi = None

        # read image list from image/binary files
        if self.opt["useContinueLearning"]:
            self.dataset_name = self.opt['dataroot_Vi'][int(self.opt["dataset_index"])].split("/")[1]
            # self.paths_Fu = common.get_image_paths(self.opt['data_type'],
            #                                        self.opt['dataroot_Fu'][int(self.opt["dataset_index"])])
            self.paths_Vi = common.get_image_paths(self.opt['data_type'],
                                                   self.opt['dataroot_Vi'][int(self.opt["dataset_index"])])
            self.paths_Ir = common.get_image_paths(self.opt['data_type'],
                                                    self.opt['dataroot_Ir'][int(self.opt["dataset_index"])])
        else:
            # self.paths_Fu = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_Fu'])
            self.paths_Vi = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_Vi'])
            self.paths_Ir = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_Ir'])
            self.dataset_name = self.opt['dataroot_Vi'].split("/")[1]

        # assert self.paths_Fu, '[Error] Fu paths are empty.'
        if self.paths_Vi and self.paths_Ir:
            assert len(self.paths_Vi) == len(self.paths_Ir), \
                '[Error] Vi: [%d] and Ir: [%d] have different number of images.' % (
                    len(self.paths_Vi), len(self.paths_Ir))

    def __getitem__(self, idx):
        if self.train:
            vi, ir, vi_path, ir_path = self._load_file(idx)
            vi = vi[:, :, 0:1]

            ir = ir[:, :, 0:1]

            vi, ir = self.get_patch1(vi, ir)
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
            # if self.transform:
            #     vi_tensor = self.transform(vi)
            #     ir_tensor = self.transform(ir)

            vi_tensor, ir_tensor = common.np2Tensor([vi, ir], self.opt['rgb_range'])
            return {'Vi': vi_tensor, 'Ir': ir_tensor, 'vi_path': vi_path,
                'ir_path': ir_path}
        else:
            EPS = 1e-8

            vi, ir, vi_path, ir_path = self._load_file(idx)

            # vi_t = torch.tensor(vi)
            # ir_t = torch.tensor(ir)

            # vi_t = vi_t.unsqueeze(dim=3)
            # ir_t = ir_t.unsqueeze(dim=3)
            # imgs = torch.concat([vi_t,ir_t], dim=3)
            # imgs = np.transpose(imgs, (3,2,0,1))
            #
            # img_cr = imgs[:, 1:2, :, :]
            # img_cb = imgs[:, 2:3, :, :]
            # w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=0)
            # w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=0)
            # fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True)
            # fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True)
            # fused_img_cr = fused_img_cr.squeeze(0)
            # fused_img_cb = fused_img_cb.squeeze(0)


            vi = vi[:, :, 0:1]
            ir = ir[:, :, 0:1]

            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
            # vi_tensor = self.transform(vi)
            # ir_tensor = self.transform(ir)

            vi_tensor, ir_tensor = common.np2Tensor([vi, ir], self.opt['rgb_range'])  #  'fused_img_cr': fused_img_cr, 'fused_img_cb': fused_img_cb,
            return {'Vi': vi_tensor, 'Ir': ir_tensor, 'vi_path': vi_path,
                'ir_path': ir_path}

    def get_patch1(self, over, under):
        h, w = over.shape[:2]
        stride = 128

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        over = over[y:y + stride, x:x + stride, :]
        under = under[y:y + stride, x:x + stride, :]

        return over, under
        # if self.train:
        #     vi, fu, ir = self._get_patch(vi, fu, ir)
        #     if self.opt["shift_pace"]:
        #         if random.random() > 0.5:
        #             pan_shift = self.dealign_ms_pan(ir.copy())
        #         else:
        #             pan_shift = ir.copy()
        #     else:
        #         pan_shift = ir.copy()
        # else:
        #     if self.opt["shift_pace"]:
        #         pan_shift = self.dealign_ms_pan(pan.copy())
        #     else:
        #         pan_shift = pan.copy()


    def __len__(self):
        if self.train:
            return 20 * len(self.paths_Vi)
            # return 2
        else:
            return len(self.paths_Vi)
            # return 1
        # return len(self.paths_Vi) * 10


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_Vi)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        vi_path = self.paths_Vi[idx]
        # fu_path = self.paths_Fu[idx]
        ir_path = self.paths_Ir[idx]
        vi = common.read_img(vi_path, self.opt['data_type'])
        # fu = common.read_img(fu_path, self.opt['data_type'])
        ir = common.read_img(ir_path, self.opt['data_type'])
        return vi, ir, vi_path, ir_path

    def _get_patch(self, vi, fu, ir):

        vi_size = self.opt['vi_size']
        # random crop and augment
        vi, fu = common.get_patch(vi, fu,
                                  vi_size, self.scale)
        vi, fu, ir = common.augment([vi, fu, ir])
        vi = common.add_noise(vi, self.opt['noise'])

        return vi, fu, ir

    def dealign_ms_pan(self, pan):
        """
        Artificially created unregistered pan and MS images
        :return:
        """
        shift_pace = random.randint(1, 250)
        h, w, c = pan.shape
        pan = np.vstack((pan[(h - shift_pace):, :], pan[:(h - shift_pace), :]))
        pan = np.hstack((pan[:, (w - shift_pace):], pan[:, :(w - shift_pace)]))
        return pan

    def random_shuffle_patch(self, pan):
        patch_width = random.randint(10, 50)
        patch_height = random.randint(10, 50)
        row_num = pan.shape[0] // patch_height
        col_num = pan.shape[1] // patch_width

        li = []
        for row in range(row_num):
            for col in range(col_num):
                li.append(pan[row * patch_height: row * patch_height + patch_width,
                          col * patch_width:col * patch_width + patch_height, :])
        np.random.shuffle(li)

        li2 = []
        for row in range(row_num):
            li2.append(li[row * col_num:row * col_num + col_num])

        li3 = []
        for item in li2:
            li3.append(np.concatenate(item, axis=1))
        li3 = np.concatenate(li3, axis=0)
        return li3
