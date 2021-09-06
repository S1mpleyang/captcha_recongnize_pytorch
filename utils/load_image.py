import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def bgr2rgb(img):
    return img[..., -1::-1].copy()


def hwc2chw(img):
    return np.transpose(img, (2, 0, 1))


def chw2hwc(img):
    return np.transpose(img, (1, 2, 0))


def img2numpy(img):
    img = to_numpy(img)
    # img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def img2tensor(img):
    """
    将HWC图片转化为CHW的tensor
    :param img:
    :return:
    """
    if img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_tensor(img)
    img = img.float()
    if img.max() > 1:
        img /= 255
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_tensor(ndarray):
    if isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def resize(img: np.ndarray, owidth, oheight):
    img = cv2.resize(img, dsize=(owidth, oheight), interpolation=cv2.INTER_LINEAR)
    return img


def load(path, resize=False, show=False):
    """
    cv2读取的图像为gbr格式，HWC格式
    :param show: 是否显示
    :param resize: 是否改变形状
    :param path:image path
    :return: a RGB image
    """
    if os.path.isfile(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_LINEAR)
        if show:
            plt.imshow(img)
            plt.show()
        return img
    else:
        print("file not exist!")
        return None


if __name__ == '__main__':
    pass