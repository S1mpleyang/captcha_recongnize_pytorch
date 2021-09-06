import torch
import torch.nn as nn

import classes


def computePrecesion(label, output):
    """

    :param label: [batch_size,4,62]
    :param output: [batch_size,4,62]
    :return:
    """
    acc_sum, n = 0, 0
    gt = label.argmax(dim=2)  # [batch_size, 4]
    pre = output.argmax(dim=2)
    flag = (gt == pre)
    n += flag.shape[0]
    for i in range(0, flag.shape[0]):
        if flag[i].int().sum().item() == 4:
            acc_sum += 1
    return acc_sum, n


def calculat_acc(output, target):
    captcha_list = classes.CLASSES
    captcha_length = 4

    output, target = output.view(-1, len(captcha_list)), target.view(-1, len(captcha_list)) # 每37个就是一个字符
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, captcha_length), target.view(-1, captcha_length) #每6个字符是一个验证码
    c = 0
    for i, j in zip(target, output):
        if torch.equal(i, j):
            c += 1
    # acc = c / output.size()[0] * 100
    acc = c
    return acc
