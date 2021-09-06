import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from utils.load_image import bgr2rgb
from utils.coconame import coco_names


def anwser_pre(pre):
    """

    :param pre: [{"boxes":tensor},{"labels":tensor},{"scores":tensor}]
    :return: boxes,labels,scores
    """
    boxes = pre[0]["boxes"].cpu().detach().numpy()
    labels = pre[0]["labels"].cpu().detach().numpy()
    scores = pre[0]["scores"].cpu().detach().numpy()
    return boxes, labels, scores


def cv2_show(
        raw_image,
        prediction,
        threshold=0.5,
        line_thickness=3,
        color=None,
        label=False,
        save_path=None,
):
    raw_image = bgr2rgb(raw_image)
    tl = line_thickness or round(0.002 * (raw_image.shape[0] + raw_image.shape[1]) / 2)
    color = color or [random.randint(0, 255) for _ in range(3)]
    boxes, labels, scores = anwser_pre(prediction)
    for i in range(len(scores)):
        if scores[i] > threshold:
            p1 = (int(boxes[i][0]), int(boxes[i][1]))
            p2 = (int(boxes[i][2]), int(boxes[i][3]))

            cv2.rectangle(
                raw_image,
                p1,
                p2,
                color=color,
                thickness=tl,
                lineType=cv2.LINE_AA,
            )
            if label:
                label = coco_names[labels[i]]
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                p2 = (p1[0] + t_size[0], p1[1] - t_size[1] - 3)

                # cv2.rectangle(raw_image, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(raw_image,
                            label,
                            (p1[0], p1[1] - 2),
                            0,
                            tl / 3,
                            [225, 255, 255],
                            lineType=cv2.LINE_AA,
                            thickness=tf)

    if save_path:
        cv2.imwrite(save_path + '/1.jpg', raw_image)


def plt_show(
        raw_image,
        prediction,
        threshold=0.5,
        line_thickness=3,
        color=None,
        label=False,
        save_path=None,
):
    tl = line_thickness or round(0.002 * (raw_image.shape[0] + raw_image.shape[1]) / 2)
    color = color or [random.randint(0, 255) for _ in range(3)]
    boxes, labels, scores = anwser_pre(prediction)
    for i in range(len(scores)):
        if scores[i] > threshold:
            p1 = (int(boxes[i][0]), int(boxes[i][1]))
            p2 = (int(boxes[i][2]), int(boxes[i][3]))

            cv2.rectangle(
                raw_image,
                p1,
                p2,
                color=color,
                thickness=tl,
                lineType=cv2.LINE_AA,
            )
            if label:
                label = coco_names[labels[i]]
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                p2 = (p1[0] + t_size[0], p1[1] - t_size[1] - 3)

                # cv2.rectangle(raw_image, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(raw_image,
                            label,
                            (p1[0], p1[1] - 2),
                            0,
                            tl / 3,
                            [225, 255, 255],
                            lineType=cv2.LINE_AA,
                            thickness=tf)
    plt.imshow(raw_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def myshow(img):
    plt.imshow(np.asarray(img))
    plt.show()
