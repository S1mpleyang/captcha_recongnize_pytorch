import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, classes=62, number=4):
        super(MyLoss, self).__init__()
        self.classes = classes
        self.number = number
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        """

        :param prediction: [batch_size, number, classes]
        :param target: [batch_size, number, classes]
        :return:
        """
        predictions = torch.flatten(predictions, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)
        loss = self.mse(predictions, targets)

        return loss.float()/predictions.shape[0]