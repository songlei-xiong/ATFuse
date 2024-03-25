import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as Gf1

from train import create_Dataloader, parse_options


def ComputeGradient(img):
    gd = np.gradient(img)
    g1 = np.sqrt(np.power(gd[0], 2) + np.power(gd[1], 2))
    return g1


def SepConvol(grad, sigma):
    v = Gf1(grad, sigma, axis=-1)
    v = Gf1(v, sigma, axis=0)
    return v


def low_pass_filter(img, sigma, niter):
    gconvolved = SepConvol(img, sigma)
    imdifference = img - gconvolved

    for i in range(0, niter):
        imconvolved = SepConvol(imdifference, sigma)
        imdifference = imdifference - imconvolved
    gconvolved = img - imdifference
    return gconvolved


def WeightingFunction(r1, r2):
    difference = r1 - r2
    ar1 = np.abs(r1)

    mask_ar = np.argwhere(ar1 <= 1)
    difference /= ar1
    difference[mask_ar[:, 0], mask_ar[:, 1]] = 0.0

    cmin = 0.25
    cmax = 0.5

    weight = (difference - cmin) / (cmax - cmin)
    mask_min = np.argwhere(difference < cmin)
    weight[mask_min[:, 0], mask_min[:, 1]] = 0.0
    mask_max = np.argwhere(difference > cmax)
    weight[mask_max[:, 0], mask_max[:, 1]] = 1

    return weight


def fast_CT(img, sigma, niter):
    grad = ComputeGradient(img)
    ratio1 = SepConvol(grad, sigma)
    gconvolved = low_pass_filter(img, sigma, niter)
    grad = ComputeGradient(gconvolved)
    ratio2 = SepConvol(grad, sigma)
    weight = WeightingFunction(ratio1, ratio2)
    return weight * gconvolved + (1 - weight) * img


if __name__ == '__main__':
    # img = imread('test.png', pilmode='F')
    #
    # sigma = 3
    # niter = 5
    #
    # material = fast_CT(img, sigma, niter)
    #
    # plt.figure(1)
    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap="gray")
    # plt.title('image')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(material, cmap="gray")
    # plt.title('material')
    #
    # texture = (img - material)
    # plt.subplot(1, 3, 3)
    # plt.imshow(texture, cmap='gray')
    # plt.title('texture')
    #
    # plt.show()

    option_file_path = '../options/train/train_MFPS_example.json'
    opt = parse_options(option_file_path)
    train_set, train_loader, val_set, val_loader = create_Dataloader(opt)
    sigma = 1
    niter = 5

    for batch in train_loader:
        images = batch
        lr = images['LR'].numpy()[0] / 2047
        hr = images['HR'].numpy()[0] / 2047
        pan = images['PAN'].numpy()[0] / 2047

        img = pan[0]

        material = fast_CT(img, sigma, niter)

        plt.figure(1)
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title('image')

        plt.subplot(1, 3, 2)
        plt.imshow(material, cmap="gray")
        plt.title('material')

        texture = (img - material)
        plt.subplot(1, 3, 3)
        plt.imshow(texture, cmap='gray')
        plt.title('texture')

        plt.show()
