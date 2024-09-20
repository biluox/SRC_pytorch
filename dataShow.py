import json

import matplotlib.pyplot as plt
import numpy as np

def average_every_ten(arr,split=10):
    # 确保输入是 NumPy 数组
    arr = np.array(arr)

    # 计算每十个数的平均数
    n = len(arr)
    averaged = []

    for i in range(0, n, split):
        # 取出每十个数
        chunk = arr[i:i+split]
        # 计算平均数并添加到结果中
        averaged.append(np.mean(chunk))

    return np.array(averaged)

if __name__ == '__main__':

    # 示例字典
    data = np.load('./dataShow/data.npz')

    # 提取数组
    x = average_every_ten(data["loss"].tolist(),1)
    y = average_every_ten(data["reward"].tolist(),1)
    z = average_every_ten(data["length"].tolist(),1)
    w = average_every_ten(data["reward_origin"].tolist(),1)

    # 绘制折线图


    plt.subplot(1,2,1)
    plt.plot(range(len(x)), x, label='loss')
    plt.plot(range(len(y)), y, label='reward')
    plt.plot(range(len(w)), w, label='reward_origin')
    # 添加标题和标签
    plt.title("loss and rewards")
    plt.xlabel("index")
    plt.ylabel("b")
    plt.legend()
    # 显示网格
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(range(len(z)), z, label='length')
    plt.title("learning path of length")
    plt.xlabel("index")
    plt.ylabel("b")
    plt.legend()
    # 显示网格
    plt.grid()


    plt.tight_layout()
# 显示图形
    plt.show()