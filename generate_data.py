from captcha.image import ImageCaptcha
import random
import os
from classes import CLASSES


def build_data(path='./testdata'):
    # build path
    img_path = os.path.join(path, "image")
    label_path = os.path.join(path, "label")
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    # generate data
    number = 5000
    start_number = len(os.listdir(img_path))
    for i in range(number):
        data, label = generate()
        save_image(start_number+i, img_path, data)
        save_label(start_number+i, label_path, label)


def save_image(name, path, data):
    image = ImageCaptcha(width=300, height=200)  # 生成指定大小的验证码照片
    image.write(data, os.path.join(path, str(name)+'.png'))  # （验证码上的信息，保存的路径）


def save_label(name, path, data):
    with open(os.path.join(path, str(name)+".txt"), 'w') as f:
        for item in data:
            f.write(str(item) + ",")


def generate():
    """

    :return: list
    """
    data = []
    label = []
    for i in range(4):
        index = random.randint(0, 61)
        label.append(index)
        data.append(CLASSES[index])
    return data, label


if __name__ == '__main__':
    build_data()
