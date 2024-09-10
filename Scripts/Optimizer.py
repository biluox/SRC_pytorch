import torch
import torch.nn as nn


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model(*data)
        training = self.model.training

        return self.criterion(output_data[1], rewards)

    def backup(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model.backup(*data)
        return self.criterion(output_data, rewards)


class ModelWithOptimizer(nn.Module):
    def __init__(self, model_with_loss, optimizer):
        super(ModelWithOptimizer, self).__init__()
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer

    def forward(self, *data):
        # 计算损失和梯度
        loss = self.model_with_loss(*data)
        training = self.model_with_loss.training
        loss.backward()  # 反向传播

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model_with_loss.parameters(), 20)

        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad()  # 清空梯度

        return loss