import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import classes
from mynetwork import Mynet, Net
from mydataset.dataset import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import os
from utils.compute_precesion import calculat_acc


def test(path):
    # 读取图片
    inputs = cv2.imread(path)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
    inputs = cv2.resize(inputs, dsize=(200, 300))
    imshow = inputs

    #     inputs = inputs.transpose(1,2,0)
    inputs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )(inputs).unsqueeze(0)

    captcha_list = classes.CLASSES
    # 加载模型
    # net = Net(62, 4)
    net = Mynet(62, 4, pretrained=False)
    model_path = '../checkpoints/resnet_transfer.pth'
    if os.path.exists(model_path):
        print('loading model...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 测试模式
    with torch.no_grad():
        outputs = net(inputs)
        outputs = outputs.view(-1, len(captcha_list))
        outputs = nn.functional.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        print(outputs)
        for i in outputs:
            print(captcha_list[i], end='')


    plt.imshow(imshow)
    plt.show()


def detect(device="cuda"):
    captcha_list = classes.CLASSES
    # model = Net(62, 4)
    model = Mynet(62, 4, pretrained=False)
    model = model.to(device)
    model_path = '../checkpoints/best.pth'
    if os.path.exists(model_path):
        print('loading model...')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    dataset = MyDataSet("../testdata", dsize=(200, 300))
    test_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    test_loop = tqdm(test_loader, leave=True)

    model.eval()
    count = 0
    TP = 0
    for batch_idx, (imgs, labels, info) in enumerate(test_loop):
        count += imgs.shape[0]
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device).view(-1, 4*62)
            outputs = model(imgs)
            TP += calculat_acc(outputs, labels)

    print("accuracy: %.5f" % (TP/count))



if __name__ == '__main__':
    # test("../testdata/image/1.png")
    detect()
