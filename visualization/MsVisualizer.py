import matplotlib.pyplot as plt
import spectral as spy
import numpy as np


def visualize_dump(ms, save_path, bands=None):
    spy.save_rgb(save_path, ms, bands=bands)


if __name__ == '__main__':
    # 获取mat格式的数据，loadmat输出的是dict，所以需要进行定位
    num = 500
    hr_ms = np.load(f"../WV2/Augment/train_HR_aug_npy/{num}_HR.npy")
    lr_ms = np.load(f"../WV2/Augment/train_LR_aug_npy/{num}_LR.npy")
    input_image_PAN = np.load(f"../WV2/Augment/train_PAN_aug_npy/{num}_PAN.npy")

    View2_lr = spy.imshow(data=lr_ms, bands=[2, 1, 0], title="img_LR")  # 图像显示
    view_pan = spy.imshow(data=input_image_PAN, title="img_pan")

    spy.save_rgb("1.bmp", hr_ms, bands=[0])

    plt.pause(60)
    k = "F:\BaiduNetdiskDownload\GF-2"
