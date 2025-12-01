from collections import defaultdict

import numpy as np


def input_data():
    """
    This function is used to take input from the user.

    Returns:
        np.ndarray: 用户输入的数据，转换为numpy数组格式
    """
    n = int(input())
    data = []
    # 读取n行数据并转换为整数列表
    for _ in range(n):
        data.append(list(map(int, input().strip().split())))
    return np.array(data)


def linear_regression():
    for k in range(K):
        y_hat = np.dot(x_train, w)
        error = y_hat - y_train
        grad = 2 / m * np.dot(x.train.T, error)
        w = w - alpha * grad

    return round(np.dot(x_test, w), 2)