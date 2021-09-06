import os
import cv2
import torch
from torch.utils.data import Dataset
from utils.load_image import load, img2tensor
import torchvision.transforms as transforms
import numpy as np

class MyDataSet(Dataset):
    def __init__(self, dir, dsize=(140,44)):
        self.img_dir = os.path.join(dir, "image")
        self.label_dir = os.path.join(dir, "label")
        self.data = os.listdir(self.img_dir)
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        self.dsize = dsize

    def __getitem__(self, item):
        name = self.data[item].split(".")[0]
        img_pth = os.path.join(self.img_dir, name + ".png")
        img = self._load_img(img_pth)
        img = self.tf(img)
        label_pth = os.path.join(self.label_dir, name + ".txt")
        label, data = self._load_label(label_pth)
        info = {"name": name, "label": data}
        return img, label, info

    def __len__(self):
        return len(self.data)

    def _load_img(self, path, **kwargs):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.dsize)
        return img

    def _load_label(self, path, **kwargs):
        with open(path, 'r') as f:
            data = f.readlines()
            data = data[0].split(',')
        data = data[0:4]
        res = torch.zeros((4, 62)).float()
        for item in range(len(data)):
            res[item][int(data[item])] = 1.0
        return res, data


if __name__ == '__main__':
    dataset = MyDataSet("../data")
    img, label, info = dataset[0]
    print(img.shape)
    print(label.shape)
    print(info)
