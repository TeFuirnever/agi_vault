import numpy as np


def distance(p1, p2):
    """
    计算两个点之间的均方差值
    
    Args:
        p1: 第一个点的坐标列表或数组
        p2: 第二个点的坐标列表或数组
        
    Returns:
        float: 两个点之间各维度差值平方的平均值
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.mean((p1 - p2) ** 2)


def knn(node_id, nodes, k):
    """
    使用K近邻算法预测目标节点的标签

    Args:
        node_id: 目标节点的ID
        nodes: 包含所有节点信息的字典
        k: 近邻数量

    Returns:
        tuple: 包含目标节点ID和预测标签的元组
    """
    # 获取目标节点的信息
    target_info = nodes[node_id]
    # 初始化一个列表用于存储距离计算结果
    res = []

    # 计算目标节点与所有其他节点之间的距离
    for idx, val in nodes.items():
        # 跳过目标节点本身
        if idx == node_id:
            continue
        # 计算目标节点与当前节点的特征距离
        dist = distance(target_info['feature'], val['feature'])
        # 将节点ID和距离存入结果列表
        res.append([idx, dist])

    # 按距离升序排列并选取前k个最近邻
    # 排序键首先按距离(x[1])排序，若距离相同则按节点ID(x[0])排序
    res = sorted(res, key=lambda x: (x[1], x[0]))
    # 取前k个最近的邻居
    res = res[:k]

    # 统计k个最近邻中各类别标签的数量
    # 提取每个最近邻节点的标签
    labels = [nodes[item[0]]['label'] for item in res]
    # 动态统计标签，而不是硬编码
    unique_labels = list(set(labels))
    stat = {label: labels.count(label) for label in unique_labels}

    # 返回数量最多的标签作为预测结果
    # 遍历统计结果，找到数量最多的标签
    if stat:  # 确保stat不为空
        max_count = max(stat.values())
        # 按标签名称排序以保证一致性
        sorted_labels = sorted([label for label, count in stat.items() if count == max_count])
        return node_id, sorted_labels[0]

    # 如果没有找到合适的标签，返回默认值-1
    return node_id, -1


def get_feature(one_node, node_dict):
    """
    获取指定节点的特征向量

    Args:
        one_node: 包含节点信息的字典
        node_dict: 包含所有节点信息的字典

    Returns:
        list: 节点的特征向量，由独热编码、邻居平均值和节点信息组成
    """
    # 获取当前节点的信息
    info = one_node['info']

    # 将info的第一个元素转换为3位二进制独热编码
    # bin()将数字转为二进制字符串，去掉'0b'前缀，用zfill(3)补齐到3位
    str_onehot = bin(info[0]).replace('0b', '').zfill(3)
    one_hot = []
    # 将二进制字符串的每一位转换为整数并加入列表
    for i in range(len(str_onehot)):
        one_hot.append(int(str_onehot[i]))

    # 计算邻居节点的信息总和
    # 初始化一个长度为4的列表用于累计邻居信息
    total = [0, 0, 0, 0]
    # 遍历当前节点的所有邻居
    for neighbor in one_node['neighbors']:
        # 获取邻居节点的信息
        neighbor_info = node_dict[neighbor]['info']
        # 累加邻居节点的信息（从索引1开始的部分）
        for i in range(len(neighbor_info[1:])):
            total[i] += neighbor_info[i]

    # 计算邻居节点信息的平均值
    # 对total中每个元素除以邻居数量得到平均值，并保留两位小数
    avg = [round(total[i] / len(one_node['neighbors']), 2) for i in range(len(total))]

    # 将独热编码、邻居平均值和节点信息（除第一个元素外）拼接成特征向量
    return one_hot + avg + info[1:]
