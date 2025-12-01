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


class DecisionTree:
    """
    决策树类，用于计算信息增益以帮助构建决策树模型
    
    Args:
        matrix (np.ndarray): 输入的数据矩阵，包含特征和标签
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def get_entropy(self, input_matrix):
        """
        计算数据集的熵值
        
        Args:
            input_matrix (np.ndarray): 输入的数据矩阵
            
        Returns:
            float: 数据的熵值，表示数据集的混乱程度，值越大表示越混乱
        """
        d = len(input_matrix)
        if d <= 0:
            return 0
        _, counts = np.unique(input_matrix[:, -1], return_counts=True)
        prob = counts / d
        no_zero_prob = prob[prob > 0]
        entropy = -np.sum(no_zero_prob * np.log2(no_zero_prob))
        return entropy if entropy else 0

    def calculate_information_gain(self, hd):
        """
        计算每个特征的信息增益
        
        Args:
            hd (float): 数据集的整体熵值
            
        Returns:
            list: 每个特征对应的信息增益值列表，信息增益越大表示该特征越重要
        """
        # 计算每个特征的信息增益
        gda = []
        # 遍历除最后一列（标签列）外的所有特征列
        for i in range(len(self.matrix[0]) - 1):
            # 根据第i个特征将数据分为两类（0和1）
            d0 = self.matrix[self.matrix[:, i] == 0]
            d1 = self.matrix[self.matrix[:, i] == 1]
            # 计算条件熵
            hda = (len(d0) * self.get_entropy(d0) + len(d1) * self.get_entropy(d1)) / len(self.matrix)
            # 计算信息增益
            gda.append(round(hd - hda, 2))
        return gda


# 6
# 1 1 0 1 1 0
# 1 0 0 1 1 1
# 0 1 0 0 1 1
# 0 1 0 1 0 0
# 0 1 0 0 0 0
# 0 0 0 1 0 0
# 获取输入数据并计算整体熵值
input_matrix = input_data()
decision_tree = DecisionTree(input_matrix)
HD = decision_tree.get_entropy(input_matrix)
GDA = decision_tree.calculate_information_gain(HD)

# 输出信息增益最大的特征索引和对应的信息增益值
if GDA:
    max_entropy = max(GDA)
    print(GDA.index(max_entropy), max_entropy)
