import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class KTDataset(Dataset):
    def __init__(self, data_folder, max_len=200):
        folder_name = os.path.basename(data_folder)
        self.dataset_name = folder_name

        # 加载数据
        with np.load(os.path.join(data_folder, folder_name + '.npz'), allow_pickle=True) as data:
            self.data = [data[k] for k in ['skill', 'y', 'real_len']]
            if folder_name == 'junyi':
                self.data[0] = data['problem']

        self.data[1] = [_.astype(np.float32) for _ in self.data[1]]

        # 计算特征数
        try:
            self.feats_num = np.max(self.data[0]).item() + 1
        except ValueError:
            self.feats_num = np.max(np.concatenate(self.data[0])).item() + 1

        self.data = list(zip(*self.data))
        self.users_num = len(self.data)
        self.max_len = max_len
        self.mask = np.zeros(self.max_len, dtype=bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        skill, y, real_len = self.data[idx]

        # 截取或填充数据
        skill = skill[:self.max_len]
        y = y[:self.max_len]

        if len(skill) < self.max_len:
            skill = np.pad(skill, (0, self.max_len - len(skill)), 'constant')
            y = np.pad(y, (0, self.max_len - len(y)), 'constant')

        # 创建掩码
        mask = self.mask.copy()
        mask[:real_len] = True

        # 转换为 PyTorch 张量
        return torch.tensor(skill, dtype=torch.int32), torch.tensor(y, dtype=torch.float32), torch.tensor(mask,
                                                                                                            dtype=torch.bool)


if __name__ == '__main__':
    dataset = KTDataset(data_folder='../datasets/assist09', max_len=200)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator())
    batch_size = 128
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 示例使用
    for batch in train_loader:
        # 处理训练批次
        print(batch)
        break