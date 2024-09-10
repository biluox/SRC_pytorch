import torch
import torch.nn.functional as F
from numpy import argsort, int32
from numpy.random import rand, randint
def pl_loss(pro, reward):
    return -torch.mean(reward * torch.log(pro + 1e-9))


# 初始生成不同的候选集，四种方案，见论文第五页 Baselines上一段话
def generate_path(batchSize, skillNum, path_type, n):
    if path_type in (0, 1):
        origin_path = argsort(rand(batchSize, n))  # 1-N concepts
        if path_type == 1:  # All concepts are grouped by size N
            origin_path += n * randint(0, skillNum // n, (batchSize, 1))
    else:  # 2 or 3
        origin_path = argsort(rand(batchSize, skillNum))  # All concepts should be sorted topN
        if path_type == 2:
            origin_path = origin_path[:, :n]  # Select N concepts randomly and sort them
    return origin_path.astype(int32)


if __name__ == '__main__':
    batch_size = 32
    skill_num = 128
    path_type = 3
    n = 10

    paths = generate_path(batch_size, skill_num, path_type, n)
    print(paths)