import numpy as np
from skimage.exposure import histogram


def hist_line_stretch(img, nbins, bound=[0.01, 0.99]):
    def _line_strectch(img):
        # img = img.astype(np.uint16)
        ori = img
        img = img.reshape(-1)
        hist1, bins1 = histogram(img, nbins=nbins, normalize=True)
        cumhist = np.cumsum(hist1)
        lowThreshold = np.where(cumhist >= bound[0])[0][0]
        highThreshold = np.where(cumhist >= bound[1])[0][0]
        lowThreshold = bins1[lowThreshold]
        highThreshold = bins1[highThreshold]
        ori[np.where(ori < lowThreshold)] = lowThreshold
        ori[np.where(ori > highThreshold)] = highThreshold
        ori = (ori - lowThreshold) / (highThreshold - lowThreshold + np.finfo(np.float).eps)
        return ori, lowThreshold, highThreshold

    if img.ndim > 2:
        lowThreshold = np.zeros(img.shape[2])
        highThreshold = np.zeros(img.shape[2])
        for i in range(img.shape[2]):
            img[:, :, i], lowThreshold[i], highThreshold[i] = _line_strectch(img[:, :, i].squeeze())
    else:
        img, lowThreshold, highThreshold = _line_strectch(img)
    return img, lowThreshold, highThreshold


def hist_line_stretchv2(img, nbins, bound=[0.01, 0.99]):
    def _line_strectch(img):
        max_img = img.max()
        min_img = img.min()
        gap_img = max_img - min_img
        gap_img = max(np.finfo(np.float).eps, gap_img)
        return (img - min_img) / gap_img, min_img, max_img

    if img.ndim > 2:
        lowThreshold = np.zeros(img.shape[2])
        highThreshold = np.zeros(img.shape[2])
        for i in range(img.shape[2]):
            img[:, :, i], lowThreshold[i], highThreshold[i] = _line_strectch(img[:, :, i].squeeze())
    else:
        img, lowThreshold, highThreshold = _line_strectch(img)
    return img, lowThreshold, highThreshold
