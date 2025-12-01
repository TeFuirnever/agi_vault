import numpy as np


def linear_regression():
    """
    实现线性回归算法，使用梯度下降法训练模型并进行预测
    
    Returns:
        np.ndarray: 测试集的预测结果，保留两位小数
    """
    # 读取模型参数
    # 从标准输入读取一行，包含5个浮点数参数
    # m: 训练样本数, n: 特征数, p: 测试样本数, alpha: 学习率, K: 迭代次数
    m, n, p, alpha, K = map(float, input().strip().split())
    # 将前4个参数转换为整数（样本数应为整数）
    m, n, p, K = int(m), int(n), int(p), int(n)

    # 读取初始权重
    # 读取n个初始权重值，这些是模型参数的初始猜测值
    w_init = list(map(float, input().strip().split()))
    # 将权重列表转换为numpy数组便于进行向量化计算
    w = np.array(w_init)

    # 读取训练数据
    # x_train: 训练特征矩阵 (m×n), y_train: 训练标签向量 (m×1)
    x_train, y_train = [], []
    # 循环读取m个训练样本
    for _ in range(m):
        # 读取一行数据，包含n个特征和1个标签
        data = list(map(float, input().strip().split()))
        # 前n个元素为特征值（输入变量）
        x_train.append(data[:-1])
        # 最后一个元素为标签值（目标变量）
        y_train.append(data[-1])
    # 转换为numpy数组以利用向量化运算
    x_train, y_train = np.array(x_train), np.array(y_train)

    # 读取测试数据
    # x_test: 测试特征矩阵 (p×n)
    x_test = []
    # 循环读取p个测试样本
    for _ in range(p):
        # 读取一行测试数据，包含n个特征（没有标签）
        data = list(map(float, input().strip().split()))
        x_test.append(data)
    # 转换为numpy数组
    x_test = np.array(x_test)

    # 使用梯度下降法训练模型
    # 进行K次迭代优化，逐步改进模型参数
    for k in range(K):
        # 计算预测值: y_hat = X * w
        # 这是线性回归的核心公式，X是训练特征矩阵，w是权重向量
        # np.dot执行矩阵乘法，结果是形状为(m,)的向量
        y_hat = np.dot(x_train, w)

        # 计算预测误差: error = y_hat - y_train
        # 即预测值与真实值的差值，也称为残差
        # 这个向量表示每个训练样本的预测偏差
        error = y_hat - y_train

        # 计算梯度: gradient = (2/m) * X^T * error
        # 这是均方误差损失函数对权重的偏导数
        # 推导过程：
        # 1. 损失函数: L = (1/m) * Σ(y_hat - y_train)²
        # 2. 对权重w求偏导: ∂L/∂w = (2/m) * X^T * (y_hat - y_train)
        # 3. 即: ∂L/∂w = (2/m) * X^T * error
        # x_train.T是x_train的转置，形状从(m,n)变为(n,m)
        # 结果gradient是一个形状为(n,)的向量，表示每个权重的梯度
        gradient = 2 / m * np.dot(x_train.T, error)

        # 更新权重: w = w - alpha * gradient
        # 沿着梯度的反方向更新权重，alpha是学习率
        # 学习率控制每次更新的步长：
        # - 太大可能导致震荡或不收敛
        # - 太小可能导致收敛速度过慢
        # 这是梯度下降法的核心更新规则
        w = w - alpha * gradient

    # 对测试集进行预测并返回结果
    # 使用训练好的权重对测试集进行预测: y_test = X_test * w
    # x_test形状为(p,n)，w形状为(n,)，结果为形状(p,)的向量
    # 结果保留两位小数
    return round(np.dot(x_test, w), 2)
