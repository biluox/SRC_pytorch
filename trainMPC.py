from argparse import ArgumentParser
import numpy as np
import os
import time

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm import tqdm

from Scripts.Envs.Env import KESEnv
from Scripts.options import get_options
from KTScripts.utils import set_random_seed
from KTScripts.DataLoader import KTDataset
# from Scripts.Envs import KESEnv
from Scripts.utils import load_agent, get_data
from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer


def main(args):
    set_random_seed(args.rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(dataset, args.model, args.dataset)
    args.skill_num = env.skill_num

    model = load_agent(args).to(device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, args.exp_name + str(args.path))
    if args.load_model:
        model.load_state_dict(torch.load(f'{model_path}'))  # 假设使用 .pth 格式
    print(f"Load Model From {model_path}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    # scheduler = LambdaLR(optimizer, lambda epoch: polynomial_decay_lr(epoch, total_epochs, args.lr, args.min_lr, 0.5))
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=True)

    criterion = BCELoss(reduction='mean')
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
                scores = env.n_step(result[0].to(device), binary=True)
                rewards = env.end_episode()
                loss = model_train(*data[:3], result, scores).cpu().detach().numpy()
                mean_reward = np.mean(rewards.cpu().detach().numpy())
                avg_time += time.perf_counter() - t0
                epoch_mean_rewards.append(mean_reward)
                all_rewards.append(mean_reward)
                print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\treward:{:.4f}'
                      .format(epoch, i, avg_time / (i + 1), loss, mean_reward))
            scheduler.step()
            print(targets[:10], '\n', result[0][:10])
            all_mean_rewards.append(np.mean(epoch_mean_rewards))
            if all_mean_rewards[-1] > best_reward:
                best_reward = all_mean_rewards[-1]
                torch.save(model.state_dict(), model_path)  # 保存模型
                print("New Best Result Saved!")
            print(f"Best Reward Now:{best_reward:.4f}")
    # else:


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'MPC', 'simulator': 'KES'})
    main(args_)
