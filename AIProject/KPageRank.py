import numpy as np


def get_top_k_recommendation(rank_vertor, k=5):
    """
    获取排名向量中前k个最高分的项目索引和分数
    
    Args:
        rank_vertor: 包含项目评分的数组
        k: 返回的推荐项目数量，默认为5
        
    Returns:
        list: 包含元组的列表，每个元组包含(索引, 分数)
    """
    # 对排名向量进行降序排序，获取排序后的索引
    # np.argsort返回升序排序的索引，[::-1]将其反转为降序
    sorted_indices = np.argsort(rank_vertor)[::-1]

    # 获取前k个索引
    # 取排序后索引的前k个元素
    top_k_indices = sorted_indices[:k]

    # 获取前k个分数
    # 使用索引数组直接获取对应的分数值
    top_k_scores = rank_vertor[top_k_indices]

    # 将索引和分数组合成元组列表返回
    # zip将两个数组对应位置的元素组合成元组
    return list(zip(top_k_indices, top_k_scores))


def page_rank_simple_sorted(alpha, matrix, iterations, k=None):
    """
    简单PageRank算法实现，返回排序后的结果

    Args:
        alpha: 阻尼因子，通常取值0.85左右
        matrix: 链接关系转移矩阵
        iterations: 迭代次数
        k: 返回前k个结果，如果为None则返回全部结果

    Returns:
        tuple: 包含排序后索引和对应分数的元组
    """
    # 获取网页数量
    # matrix.shape[0]返回矩阵的第一维度大小，即行数（网页数量）
    n = matrix.shape[0]

    # 初始化PageRank值，均匀分布
    # np.ones(n)创建一个包含n个1的数组，除以n得到均匀分布的初始值
    rank = np.ones(n) / n

    # 进行多次迭代计算PageRank值
    # 迭代iterations次，逐步优化PageRank值
    for _ in range(iterations):
        # PageRank核心公式: PR = α * M * PR + (1-α) / N
        # 这是PageRank算法的矩阵形式：
        # - M是转移概率矩阵
        # - PR是当前PageRank值向量
        # - α是阻尼因子（通常为0.85）
        # - (1-α)/N是随机跳转部分
        # - @表示矩阵乘法
        rank = alpha * matrix @ rank + (1 - alpha) / n

    # 对最终的PageRank值进行降序排序
    # 获取按PageRank值降序排列的索引
    sorted_indices = np.argsort(rank)[::-1]
    # 根据排序索引获取对应的PageRank值
    sorted_scores = rank[sorted_indices]

    # 根据k值决定返回多少个结果
    if k is not None and k < n:
        # 如果指定了k且k小于网页总数，则返回前k个结果
        return sorted_indices[:k], sorted_scores[:k]
    else:
        # 否则返回所有结果
        return sorted_indices, sorted_scores
