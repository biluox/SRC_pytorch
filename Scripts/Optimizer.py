import torch
import torch.nn as nn
import torchviz


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model(*data)
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
        loss = self.model_with_loss.backup(*data)
        loss.backward()  # 反向传播
        dot = torchviz.make_dot(loss, params=dict(self.model_with_loss.named_parameters()))
        dot.render("model_graph", format="png")  # 保存为 PNG 文件

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model_with_loss.parameters(), 20)

        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad()  # 清空梯度

        return loss