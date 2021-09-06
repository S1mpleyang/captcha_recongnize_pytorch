def testimg():
        import cv2
        img = cv2.imread("./data/image/0.png")
        print(img.shape)
        cv2.imshow("show", img)
        cv2.waitKey(1000)

def testweight():
        import torch
        from model.mynetwork import Mynet
        # model1 = Mynet(62, 4)
        # print(model1.state_dict())


        model2 = Mynet(62, 4)
        model2.load_state_dict(torch.load("./model/myweight.pth"))
        print(model2.state_dict())

def test1():
        import torch
        mse = torch.nn.MSELoss(reduction="sum")


        a = torch.tensor(
        [
                [
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                #        [1, 0, 0, 0],
                ],
                [
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        #        [1, 0, 0, 0],
                ],
                [
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        [2, 0, 0, 0, 2, 0, 0, 0],
                        # [1, 0, 0, 0],
                ],
        ],
                dtype=float)
        b = torch.tensor(
        [
                [
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        # [0.5, 0.1, 0.2, 0.1],
                ],
                [
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        # [0.5, 0.1, 0.2, 0.1],
                ],
                [
                        [1, 0, 0, 1, 2, 0, 0, 0],
                        [1, 0, 0, 1, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        [1, 0, 0, 0, 2, 0, 0, 0],
                        # [0.5, 0, 0, 0],
                ],
        ],
                dtype=float)

        print(a.shape)
        print(b.shape)
        print(a-b)
        print(mse(a[..., 0, ...],b[..., 0, ...]))
        print(mse(a[..., 1, ...], b[..., 1, ...]))
        print(mse(a,b))
        # c = b
        # b = b.argmax(dim=2)
        # c = c.argmax(dim=2)
        # y = (c==b)
        # print(y.shape)
        # print(y)
        # print(y[2].int().sum().item())

def test3():
        from torch.autograd import gradcheck
        import torch
        import torch.nn as nn
        import torch
        from model.mynetwork import Mynet
        # model1 = Mynet(62, 4)
        # print(model1.state_dict())


        model2 = Mynet(62, 4).double()

        # 定义神经网络模型
        class Net(nn.Module):

                def __init__(self):
                        super(Net, self).__init__()
                        self.net = nn.Sequential(
                                nn.Linear(15, 30),
                                nn.ReLU(),
                                nn.Linear(30, 15),
                                nn.ReLU(),
                                nn.Linear(15, 1),
                                nn.Sigmoid()
                        )

                def forward(self, x):
                        y = self.net(x)
                        return y

        net = Net()
        net = net.double()
        inputs = torch.randn((1,3,200,300), requires_grad=True, dtype=float)
        test = gradcheck(model2, inputs)
        print("Are the gradients correct: ", test)


if __name__ == '__main__':
    test3()
