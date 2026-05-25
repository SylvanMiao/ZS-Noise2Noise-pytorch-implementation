import torch
import torch.nn.functional as F


# 下采样
def pair_downsampler(img):
    #img has shape B C H W
    if not torch.is_floating_point(img):
        img = img.float()
    c = img.shape[1]

    filter1 = torch.tensor([[[[0, 0.5], [0.5, 0]]]], device=img.device, dtype=img.dtype)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.tensor([[[[0.5, 0], [0, 0.5]]]], device=img.device, dtype=img.dtype)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2