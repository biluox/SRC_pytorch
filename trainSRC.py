import json
import os
import signal
import sys
import time
from argparse import ArgumentParser

import keyboard
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import set_random_seed
from Scripts.Agent.utils import pl_loss
from Scripts.Envs.Env import KESEnv
from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer
from Scripts.options import get_options
from Scripts.utils import load_agent, get_data
from trainDKT import polynomial_decay_lr

dataShow = {'loss': [], 'reward': [], 'length': [], 'reward_origin': []}


def main(args):
    set_random_seed(args.rand_seed)
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(dataset, args.model, args.dataset, device)
    args.skill_num = env.skill_num
    model = load_agent(args).to(device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, args.exp_name + str(args.path))

    if args.load_model:
        model.load_state_dict(torch.load(f'{model_path}'))  # 假设使用 .pth 格式
        print(f"Load Model From {model_path}")
    total_epochs = args.num_epochs  # 假设已有总训练周期
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    # scheduler = LambdaLR(optimizer, lambda epoch: polynomial_decay_lr(epoch, total_epochs, args.lr, args.min_lr, 0.5))
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=True)

    criterion = pl_loss

    model_with_loss = ModelWithLoss(model, criterion)
    model_train = ModelWithOptimizer(model_with_loss, optimizer)
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    targets, result = None, None
    best_reward = -1e9
    if args.isTrain:
        print('-' * 20 + "Training Start" + '-' * 20)
        model_train.train()
        model_with_loss.train()
        model.train()

        for epoch in range(args.num_epochs):
            avg_time = 0
            epoch_mean_rewards = []
            for i in tqdm(range(200)):
                t0 = time.perf_counter()
                targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
                initial_log_scores = env.begin_episode(targets, initial_logs)
                data = (
                    targets.to(device), initial_logs.to(device), initial_log_scores.to(device), origin_path.to(device),
                    args.steps)
                result = model(*data)
                length = torch.count_nonzero(result[4], dim=1)
                mean_length = torch.mean(length.detach().float()).item()
                env.n_step(result[4].to(device), binary=True)
                rewards, reward_punish = env.end_episode(length)
                loss = model_train(*data[:-1], result[2], reward_punish).cpu().detach().numpy()  # 和原文不一样
                mean_reward = np.mean(reward_punish.cpu().detach().numpy())
                mean_reward_origin = np.mean(rewards.cpu().detach().numpy())
                avg_time += time.perf_counter() - t0
                epoch_mean_rewards.append(mean_reward)
                all_rewards.append(mean_reward)
                dataShow['loss'].append(loss.item())
                dataShow['reward'].append(mean_reward)
                dataShow['reward_origin'].append(mean_reward_origin)
                dataShow['length'].append(mean_length)
                print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\treward:{:.4f}\tlength:{}'
                      .format(epoch, i, avg_time / (i + 1), loss, mean_reward, mean_length))
            scheduler.step()
            print(targets[:10], '\n', result[0][:10])
            all_mean_rewards.append(np.mean(epoch_mean_rewards))
            if all_mean_rewards[-1] > best_reward:
                best_reward = all_mean_rewards[-1]
                torch.save(model.state_dict(), model_path)  # 保存模型
                print("New Best Result Saved!")
            print(f"Best Reward Now:{best_reward:.4f}")
    else:
        print('-' * 20 + "Testing Start" + '-' * 20)
        test_rewards = []
        model_with_loss.eval()
        model.load_state_dict(torch.load(f'{model_path}'))  # 假设使用 .pth 格式
        for i in tqdm(range(200)):
            targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets.to(device), initial_logs.to(device), initial_log_scores.to(device), origin_path.to(device),
                    args.steps)
            result = model(*data)
            env.n_step(result[0], binary=True)
            rewards = env.end_episode()
            loss = criterion(result[1], rewards).cpu().detach().numpy()  # 和原文不一样
            mean_reward = np.mean(rewards.cpu().detach().numpy())
            dataShow['loss'].append(loss.item())
            dataShow['reward'].append(mean_reward)
            test_rewards.append(mean_reward)
            print(f'batch:{i}\tloss:{loss:.4f}\treward:{mean_reward:.4f}')
        print(result[0][:10])
        print(f"Mean Reward for Test:{np.mean(test_rewards)}")


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'SRC', 'simulator': 'KES'})


    def cleanup():
        print("运行结束了，保存数据")
        dataShow["loss"] = np.array(dataShow["loss"])
        dataShow["reward"] = np.array(dataShow["reward"])
        dataShow["length"] = np.array(dataShow["length"])
        dataShow["reward_origin"] = np.array(dataShow["reward_origin"])
        np.savez('./dataShow/data.npz', loss=dataShow["loss"], reward=dataShow["reward"], length=dataShow["length"],
                 reward_origin=dataShow["reward_origin"])


    try:
        main(args_)
    finally:
        cleanup()
