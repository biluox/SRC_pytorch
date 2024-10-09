import argparse
import os
import torch
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from Scripts.Envs.Env import KESEnv
from Scripts.Envs.utils import load_d_agent
from Scripts.utils import get_data


def rank(env, path):
    # 将 path 转换为 PyTorch 张量并重塑
    path = path.reshape(-1, 1)
    path_tensor = torch.tensor(path, dtype=torch.int32)  # 假设 path 是浮点数类型

    # 获取数据
    targets, initial_logs, _ = get_data(path_tensor.shape[0], env.skill_num, 3, 10, 0, 1)

    # 开始新的一集
    env.begin_episode(targets, initial_logs)

    # 执行 n 步
    env.n_step(path_tensor)

    # 结束一集并获取奖励
    rewards = env.end_episode().squeeze(-1)

    # 使用 PyTorch 的排序功能
    sorted_indices = torch.argsort(rewards, descending=False)  # 降序排序
    ranked_path = path_tensor.reshape(-1)[sorted_indices]

    return ranked_path


def testRuleBased(args):
    # 加载数据集和环境
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(dataset, args.model, args.dataset)

    # 获取数据
    targets, initial_logs, origin_paths = get_data(args.batch, env.skill_num, 3, 10, args.p, args.n)

    ranked_paths = [None] * args.batch

    if args.agent == 'rule':
        for i in tqdm(range(args.batch)):
            path = rank(env, origin_paths[i])[-args.n:]
            ranked_paths[i] = path
        ranked_paths = torch.stack(ranked_paths)  # 使用 PyTorch 的 stack 函数

    elif args.agent == 'random':
        ranked_paths = origin_paths[:, -args.n:]

    elif args.agent == 'GRU4Rec':
        d_model = load_d_agent(args.agent, args.dataset, env.skill_num, False)
        ranked_paths = d_model.GRU4RecSelect(origin_paths, args.n, env.skill_num, initial_logs)

    print(ranked_paths)

    # 开始新的 episode
    env.begin_episode(targets, initial_logs)

    # 执行 n 步
    env.n_step(ranked_paths)

    # 打印平均奖励
    print(torch.mean(env.end_episode()))


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='./datasets')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='assist09', choices=['assist09', 'junyi', 'assist15'])
    parser.add_argument('-a', '--agent', type=str, default='GRU4Rec', choices=['rule', 'random', 'GRU4Rec'])
    parser.add_argument('-m', '--model', type=str, default='DKT', choices=['DKT', 'CoKT'])
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('-p', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args_ = parser.parse_args()
    testRuleBased(args_)
