import os
import math

import torch
from scipy.stats import pearsonr
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from torch.autograd import Variable

eps = torch.finfo(torch.float32).eps


####################
# miscellaneous
####################

def var2device(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
def Tensor2np(tensor_list, rgb_range):
    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint16)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def quantize(img, rgb_range):
    pixel_range = 2047. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 2047).round()


def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    C, _, _ = img1.shape
    img1 = img1.reshape(C, -1)
    img2 = img2.reshape(C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
            eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)


###Pan-sharpening#####
def calc_metrics(imgFu,img1, img2, crop_border, test_Y=True):
    RMSE1 = np.sqrt(((imgFu - img1) ** 2).mean())
    RMSE2 = np.sqrt(((imgFu - img2) ** 2).mean())
    RMSE = 0.5 * RMSE1 + 0.5 * RMSE2
    # Qabf = qabfMetric(imgFu, img1, img2)
    CC1 = cc(imgFu, img1)
    CC2 = cc(imgFu, img2)
    CC = 0.5 * CC1 + 0.5 * CC2
    return CC, RMSE


####################
# metric
####################
def calc_metrics_(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height - crop_border, crop_border:width - crop_border, :]
        cropped_im2 = im2_in[crop_border:height - crop_border, crop_border:width - crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height - crop_border, crop_border:width - crop_border]
        cropped_im2 = im2_in[crop_border:height - crop_border, crop_border:width - crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')










#if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
#如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)




# 数组旋转180度
def flip180(arr):

    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

#相当于matlab的Conv2
def convolution(k, data):
    k = flip180(k)
    data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    n,m = data.shape
    img_new = []
    for i in range(n-2):
        line = []
        for j in range(m-2):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)


#用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
#matlab会对图像进行补0，然后卷积核选择180度
#gA = sqrt(SAx.^2 + SAy.^2);
#定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
def getArray(img):
    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)



    SAx = convolution(h3,img)
    SAy = convolution(h1,img)
    gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
    n, m = img.shape
    aA = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(SAx[i,j]==0):
                aA[i,j] = math.pi/2
            else:
                aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
    return gA,aA



#the relative strength and orientation value of GAF,GBF and AAF,ABF;
def getQabf(aA,gA,aF,gF):
    L = 1;
    Tg = 0.9994;
    kg = -15;
    Dg = 0.5;
    Ta = 0.9879;
    ka = -22;
    Da = 0.8;
    n, m = aA.shape
    GAF = np.zeros((n,m))
    AAF = np.zeros((n,m))
    QgAF = np.zeros((n,m))
    QaAF = np.zeros((n,m))
    QAF = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if(gA[i,j]>gF[i,j]):
                GAF[i,j] = gF[i,j]/gA[i,j]
            elif(gA[i,j]==gF[i,j]):
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i,j]/gF[i, j]
            AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

            QgAF[i,j] = Tg/(1+math.exp(kg*(GAF[i,j]-Dg)))
            QaAF[i,j] = Ta/(1+math.exp(ka*(AAF[i,j]-Da)))

            QAF[i,j] = QgAF[i,j]*QaAF[i,j]

    return QAF

# QAF = getQabf(aA,gA,aF,gF)
# QBF = getQabf(aB,gB,aF,gF)
#
#
# #计算QABF
# deno = np.sum(gA+gB)
# nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))
# output = nume/deno

def qabfMetric(imgFu, imgA, imgB):
    gA, aA = getArray(imgA)
    gB, aB = getArray(imgB)
    gF, aF = getArray(imgFu)
    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output

