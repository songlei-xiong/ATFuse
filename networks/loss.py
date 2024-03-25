
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF




class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused, thresholds):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        # gradient_joint = torch.max(gradient_A, gradient_B)
        grant_joint = torch.concat([gradient_A, gradient_B], dim=1)
        grant_joint_max, index = grant_joint.max(dim=1)

        a, b, c, d = gradient_A.size(0), gradient_A.size(1), gradient_A.size(2), gradient_A.size(3)
        grant_joint_max = grant_joint_max.reshape(a, b, c, d)

        gradient_A_Mask = threshold_tensor(gradient_A, dim=2, k=thresholds)
        aaa = gradient_A_Mask.argmax(dim=1).shape
        gradient_B_Mask = threshold_tensor(gradient_B, dim=2, k=thresholds)
        bbb = gradient_B_Mask.argmax(dim=1).shape



        Loss_gradient = F.l1_loss(gradient_fused, grant_joint_max)
        return Loss_gradient, gradient_A_Mask, gradient_B_Mask



def gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A, gradient_B, L_Inten_loss, percent, mask_pre = None):
    """
    percent:百分比，大于百分之多少的像素点
    L_Inten_loss：计算像素损失的函数
    gradient_A:A图像的梯度
    mask_pre:前一次的掩膜，第一次前百分之20，第二次取60，就是中间的四十
    """
    thresholds = torch.round(torch.tensor(percent * image_A_Y.shape[2] * image_A_Y.shape[3])).int()
    clone_grand_A = gradient_A.clone().detach()
    gradient_A_Mask = threshold_tensor(clone_grand_A, dim=2, k=thresholds)


    clone_grand_B = gradient_B.clone().detach()
    gradient_B_Mask = threshold_tensor(clone_grand_B, dim=2, k=thresholds)

    if mask_pre == None:
        grand_Mask = gradient_A_Mask + gradient_B_Mask
        grand_Mask = grand_Mask.clamp(min=0, max=1)

    else:
        grand_Mask = gradient_A_Mask + gradient_B_Mask
        grand_Mask = grand_Mask.clamp(min=0, max=1)

        grand_Mask -= mask_pre
    grand_IntenLoss = L_Inten_loss(image_A_Y * grand_Mask, image_B_Y * grand_Mask, image_fused_Y * grand_Mask)
    return grand_IntenLoss, grand_Mask


def testNum(grand_Mask):
    grand_Mask_1Wei = torch.flatten(grand_Mask)
    num = 0
    for i in range(grand_Mask_1Wei.shape[0]):
        if grand_Mask_1Wei[i] == 1:
            num += 1
    return num

class L_Grad_Inte(nn.Module):
    """
        按梯度分块求像素损失并计算梯度损失
    """
    def __init__(self):
        super(L_Grad_Inte, self).__init__()
        self.sobelconv=Sobelxy()
        self.L_Inten_aver = L_IntensityAver()
        self.L_Inten_Max = L_Intensity()
        self.L_Inten_Once = L_IntensityOnce()
    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_fused = self.sobelconv(image_fused_Y)
        grant_joint = torch.concat([gradient_A, gradient_B], dim=1)
        grant_joint_max, index = grant_joint.max(dim=1)

        a, b, c, d = gradient_A.size(0), gradient_A.size(1), gradient_A.size(2), gradient_A.size(3)
        grant_joint_max = grant_joint_max.reshape(a, b, c, d)

#梯度乘以像素强度来对图像进行分等级求强度loss
        gradient_A_Att = image_A_Y * gradient_A
        gradient_B_Att = image_B_Y * gradient_B


#前百分之20的梯度的像素点用max像素损失
        grand_IntenLoss_one, grand_Mask_one = gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A_Att, gradient_B_Att, self.L_Inten_Max, 0.8, mask_pre = None)
# #百分之20-70的用平均
#         grand_IntenLoss_two, grand_Mask_two = gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A_Att, gradient_B_Att, self.L_Inten_aver, 0.3, mask_pre = grand_Mask_one)
# 最后30用vi的像素点
        grand_Mask_three = 1 - grand_Mask_one
        grand_IntenLoss_three = self.L_Inten_aver(image_A_Y * grand_Mask_three, image_B_Y * grand_Mask_three, image_fused_Y * grand_Mask_three)

        grand_IntenLoss = grand_IntenLoss_one + grand_IntenLoss_three

        Loss_gradient = F.l1_loss(gradient_fused, grant_joint_max)
        return Loss_gradient, grand_IntenLoss








def threshold_tensor(input_tensor, dim, k):
    """
    将输入的Tensor按维度dim取第k大的元素作为阈值，大于等于阈值的元素置为1，其余元素置为0。

    Args:
    - input_tensor: 输入的Tensor
    - dim: 取第k大元素的维度
    - k: 取第k大元素

    Returns:
    - 输出的Tensor，形状与输入的Tensor相同
    """
    # kth_value, _ = torch.kthvalue(input_tensor, k, dim=dim, keepdim=True)  # 按维度dim取第k大的元素
    B, N, C ,D = input_tensor.shape
    input_tensor = input_tensor.reshape(B,N,C*D)
    for i in range(B):
        kth_value, _ = torch.kthvalue(input_tensor[i:i+1, :, :], k, dim=dim, keepdim=True)
        kth_value = torch.flatten(kth_value)
        input_tensor[i:i+1,: , :] = torch.where(input_tensor[i:i+1, :, :] >= kth_value[0], torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    input_tensor = input_tensor.reshape(B, N, C ,D)
    return input_tensor


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(intensity_joint, image_fused)
        return Loss_intensity


class L_IntensityAver(nn.Module):
    def __init__(self):
        super(L_IntensityAver, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        Loss_intensity_A = F.l1_loss(image_A, image_fused)
        Loss_intensity_B = F.l1_loss(image_B, image_fused)
        Loss_intensity = 0.5 * Loss_intensity_A + 0.5 * Loss_intensity_B
        return Loss_intensity


class L_IntensityOnce(nn.Module):
    def __init__(self):
        super(L_IntensityOnce, self).__init__()

    def forward(self, image_A, image_fused):

        Loss_intensity = F.l1_loss(image_A, image_fused)
        return Loss_intensity


class L_Intensity_GrandFu(nn.Module):
    def __init__(self):
        super(L_Intensity_GrandFu, self).__init__()

    def forward(self,image_A, image_B, image_fused, gradient_A_Mask, gradient_B_Mask):

        Fu_image_maskA_A = image_A * gradient_A_Mask
        Loss_intensity_maskA = F.l1_loss(image_fused * gradient_A_Mask, Fu_image_maskA_A)


        Fu_image_maskB_B = image_B * gradient_B_Mask
        Loss_intensity_maskB = F.l1_loss(image_fused * gradient_B_Mask, Fu_image_maskB_B)

        return Loss_intensity_maskA + Loss_intensity_maskB



class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_GradInte = L_Grad_Inte()

        # print(1)
    def forward(self, image_fused, image_A, image_B):

        image_fused = image_fused["pred"]

        loss_gradient, grand_IntenLoss = self.L_GradInte(image_A, image_B, image_fused)

        fusion_loss = loss_gradient * 20 + grand_IntenLoss * 20

        return fusion_loss
