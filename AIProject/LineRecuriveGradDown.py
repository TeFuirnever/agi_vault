import numpy as np


def cal_grad(n, train_data, y, learning_rate, w0, m):
    """
    计算线性回归模型的梯度
    
    Args:
        n: 特征数量
        train_data: 训练数据集，形状为(m, n)
        y: 训练标签，形状为(m,)
        learning_rate: 学习率参数
        w0: 当前权重向量，形状为(n,)
        m: 训练样本数量
        
    Returns:
        np.ndarray: 梯度向量，形状为(n,)
    """
    # 计算预测值
    predictions = np.dot(train_data, w0)
    # 计算预测值与真实值之间的误差
    errors = predictions - y
    # 计算梯度并乘以学习率
    gradient = (2 / m) * np.dot(errors, train_data) * learning_rate
    return gradient


def func(n, m, p, learning_rate, K, w0, train_data, test_data, y):
    """
    执行线性回归梯度下降算法并进行预测
    
    Args:
        n: 特征数量
        m: 训练样本数量
        p: 测试样本数量
        learning_rate: 学习率
        K: 迭代次数
        w0: 初始权重向量
        train_data: 训练数据集，形状为(m, n)
        test_data: 测试数据集，形状为(p, n)
        y: 训练标签，形状为(m,)
        
    Returns:
        list: 测试集预测结果列表
    """
    # 执行K次梯度下降迭代
    for _ in range(K):
        grad = cal_grad(n, train_data, y, learning_rate, w0, m)
        w0 -= grad
    # 使用训练好的模型对测试集进行预测
    py = np.dot(test_data, w0)
    return py.tolist()
