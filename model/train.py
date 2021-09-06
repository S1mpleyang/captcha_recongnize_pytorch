from mynetwork import Mynet, Net
from mydataset.dataset import MyDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.compute_precesion import computePrecesion
from loss import MyLoss
import torch.nn as nn
from torch.autograd import Variable, gradcheck
from torch.utils.tensorboard import SummaryWriter
import os
from utils.compute_precesion import calculat_acc


def test_fn(test_loader, device, model):
    test_loop = tqdm(test_loader, leave=True)
    count = 0
    TP = 0
    for batch_idx, (imgs, labels, info) in enumerate(test_loop):
        count += imgs.shape[0]
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device).view(-1, 4 * 62)
            outputs = model(imgs)
            TP += calculat_acc(outputs, labels)

    return TP / count


def train_fn(train_loop, model, optimizer, criterion, device):
    running_loss = 0.0
    acc = 0.0
    count = 0
    for batch_idx, (imgs, labels, info) in enumerate(train_loop):
        count += imgs.shape[0]
        imgs = imgs.to(device)  # img shape= [batch_size, C, H, W]
        labels = labels.to(device).view(-1, 4 * 62)  # label shape= [batch_size, 4 * 62]

        optimizer.zero_grad()
        outputs = model(imgs)  # output shape= [batch_size, 4 * 62]
        # compute loss

        # print(outputs.shape)
        # print(labels.shape)
        train_loss = criterion(outputs, labels)

        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.item()
        acc += calculat_acc(outputs, labels)

    return running_loss / count, acc / count


def train(
        device="cuda",
        batch_size=128,
        learning_rate=1e-2,
        epochs=200,
        save=True,
):
    model = Net(62, 4)
    # model = Mynet(62, 4, pretrained=False)
    model = model.to(device)
    criterion = nn.MultiLabelSoftMarginLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 优化器
    start_epoch = 0

    # 加载模型
    model_path = '../checkpoints/model.pth'
    if os.path.exists(model_path):
        print('loading model...')
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_dataset = MyDataSet("../data", dsize=(140, 44))
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
    )
    detect_dataset = MyDataSet("../testdata", dsize=(140, 44))
    test_loader = DataLoader(
        detect_dataset,
        shuffle=True,
        batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    tensorboard_writer = SummaryWriter("runs/trainlog")

    best_pth = {
        'epoch': 0,
        'accuracy': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # start training
    for epoch in range(epochs):
        model.train()
        train_loop = tqdm(train_loader, leave=True)
        mean_loss, train_acc = train_fn(train_loop, model, optimizer, criterion, device)

        # write to tensorboard
        tensorboard_writer.add_scalar("loss", mean_loss, epoch + 1 + start_epoch)
        tensorboard_writer.add_scalar("train_accuracy", train_acc, epoch + 1 + start_epoch)
        tensorboard_writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], epoch + 1 +start_epoch)

        # 结束一个epoch,计算测试集的正确率
        model.eval()
        test_acc = test_fn(test_loader, device, model)
        tensorboard_writer.add_scalar("test_accuracy", test_acc, epoch + 1 + start_epoch)

        # save best weight
        if test_acc > best_pth['accuracy']:
            best_pth['epoch'] = epoch
            best_pth['accuracy'] = test_acc
            best_pth['model_state_dict'] = model.state_dict()
            best_pth['optimizer_state_dict'] = optimizer.state_dict()

        # print result
        if epoch % 20 == 0:
            print("epoch=%s, loss=%.10f, train_accuracy=%.5f, test_accuracy=%.5f"
                  % (epoch, mean_loss, train_acc, test_acc))

        # 每50个epoch 更新学习率
        if (epoch+1) % 50 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9

    # save model
    if save:
        torch.save(
            {
                'epoch': start_epoch + epochs,
                'accuracy': test_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
        # torch.save(best_pth, '../checkpoints/net1best.pth')
    tensorboard_writer.close()


if __name__ == '__main__':
    train()
