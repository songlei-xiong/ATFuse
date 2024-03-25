import warnings

import torch
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


if __name__ == '__main__':
    B, C, W, H = 2, 3, 1024, 1024
    x = torch.randn(B, C, H, W)

    kernel_size = 128
    stride = 64
    patches = x.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride)
    print(patches.shape)  # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]

    # perform the operations on each patch
    # ...

    # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C, -1, kernel_size * kernel_size)
    print(patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)
    print(patches.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(B, C * kernel_size * kernel_size, -1)
    print(patches.shape)  # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    output = F.fold(
        patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)
    print(output.shape)  # [B, C, H, W]
