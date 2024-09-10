import glob
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def load_csv_file(file):
    return np.loadtxt(file, skiprows=1, dtype=np.int32, delimiter=',', ndmin=2)


def deal_assist15(data_dir):
    raw_dir = os.path.join(data_dir, 'assist15', 'processed', '1')
    files = glob.glob(os.path.join(raw_dir, '**/*.csv'), recursive=True)

    with Pool(8) as pool:
        logs = list(tqdm(pool.imap(load_csv_file, files), total=len(files)))

    print(len(logs), '\n', logs[0])
    real_lens = [len(_) for _ in logs]
    logs = np.concatenate(logs)
    logs[:, 0] = np.unique(logs[:, 0], return_inverse=True)[1]
    skills, ys = logs[:, 0], logs[:, 1]
    skills = np.split(skills, np.cumsum(real_lens)[:-1])
    ys = np.split(ys, np.cumsum(real_lens)[:-1])
    save_dir = {'skill': np.asarray(skills, dtype=object), 'y': np.asarray(ys, dtype=object), 'real_len': real_lens}
    np.savez(os.path.join(data_dir, 'assist15', 'assist15.npz'), **save_dir)


if __name__ == '__main__':
    deal_assist15('./data')
