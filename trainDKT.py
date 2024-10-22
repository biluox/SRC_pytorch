import os
import time
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from KTScripts.PredictModel import ModelWithLoss
from KTScripts.options import get_options
from KTScripts.utils import set_random_seed, load_model, evaluate_utils
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

def dataSPlit(dataset, train_num=0.8):
    train_size = int(train_num * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator())
    batch_size = args_.batch_size
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 自定义多项式衰减学习率
def polynomial_decay_lr(epoch, total_epochs, initial_lr, end_lr, power):
    if epoch >= total_epochs:
        return end_lr
    else:
        decay = (1 - epoch / total_epochs) ** power
        return initial_lr + (end_lr - initial_lr) * decay


patient = 0
def main(args: Namespace):
    set_random_seed(args.rand_seed)
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    args.feat_nums, args.user_nums = dataset.feats_num, dataset.users_num
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    train_loader, test_loader = dataSPlit(dataset)
    if args.forRec:
        args.output_size = args.feat_nums
    model = load_model(args).to(device)

    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 加载模型
    if args.load_model:
        model.load_state_dict(torch.load(f'{model_path}'))  # 假设使用 .pth 格式
        print(f"Load Model From {model_path}")

    # 初始化学习率、优化器和损失函数
    total_epochs = args.num_epochs  # 假设已有总训练周期
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = LambdaLR(optimizer, lambda epoch: polynomial_decay_lr(epoch, total_epochs, args.lr, args.min_lr, 0.5))

    model_with_loss = ModelWithLoss(model, nn.BCELoss(reduction='mean'))
    best_val_auc = 0

    train_total, test_total = len(train_loader), len(test_loader)
    print('-' * 20 + "Training Start" + '-' * 20)

    for epoch in range(args.num_epochs):
        avg_time = 0
        model.train()  # 设置模型为训练模式
        print(f'第{epoch + 1}轮')
        for i, data in tqdm(enumerate(train_loader), total=train_total):
            t0 = time.perf_counter()
            # 假设 data 是一个包含输入和标签的元组
            inputs, labels,mask = data  # 根据实际数据结构解包
            inputs, labels,mask = inputs.to(device) , labels.to(device), mask.to(device)
            optimizer.zero_grad()  # 清空梯度
            # data =
            # 前向传播
            loss, output_data = model_with_loss(inputs, labels,mask)  # 假设模型封装了损失计算
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 计算损失和评估指标
            avg_time += time.perf_counter() - t0
            acc, auc = evaluate_utils(*output_data)  # 假设 evaluate_utils 已定义
            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\tacc:{:.4f}\tauc:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss.item(), acc, auc))
        scheduler.step()
        # print('-' * 20 + "Validating Start" + '-' * 20)
        val_eval = [[], []]
        loss_total, data_total = 0, 0
        model.eval()  # 设置模型为评估模式

        with torch.no_grad():  # 禁用梯度计算
            for data in tqdm(test_loader, total=test_total):  # 假设 test_loader 已定义

                inputs, labels,mask = data  # 根据实际数据结构解包
                inputs, labels,mask = inputs.to(device) , labels.to(device), mask.to(device)
                loss, output_data = model_with_loss.output(inputs, labels,mask)  # 获取损失和输出

                val_eval[0].append(output_data[0].cpu().numpy())  # 将数据移到 CPU 并转换为 NumPy
                val_eval[1].append(output_data[1].cpu().numpy())
                loss_total += loss.item() * len(data[0])
                data_total += len(data[0])

        val_eval = [np.concatenate(_) for _ in val_eval]
        acc, auc = evaluate_utils(*val_eval)
        print(f"Validating loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")

        if auc >= best_val_auc:
            patient=0
            best_val_auc = auc
            torch.save(model.state_dict(), model_path)  # 保存模型
            print("New best result Saved!")
        else:
            patient +=1
            if patient > 100:
                print('训练已经到顶' + '-'*20)
                break

        print(f"Best Auc Now:{best_val_auc:.4f}")

if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser)
    main(args_)
