import torch
from classes import CLASSES



def text2vec(text, captcha_length=4, captcha_list=CLASSES):
    """
    # 验证码文本转为向量

    :param text:
    :param captcha_length:
    :param captcha_list:
    :return:
    """
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len > captcha_length:
        raise ValueError("验证码超过4位啦！")
    for i in range(text_len):
        vector[i, captcha_list.index(text[i])] = 1
    return vector


def vec2text(vec, captcha_list=CLASSES):
    """
    # 验证码向量转为文本

    :param captcha_list:
    :param vec:
    :return:
    """
    label = torch.nn.functional.softmax(vec, dim=1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [captcha_list[v] for v in vec]
    return ''.join(text_list)
