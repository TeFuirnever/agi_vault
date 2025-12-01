def min_max_value(xi, x_min, x_max):
    """
    对单个值进行最小-最大归一化

    Args:
        xi: 需要归一化的值
        x_min: 数据集中的最小值
        x_max: 数据集中的最大值

    Returns:
        float: 归一化后的值，范围在[0,1]之间
    """
    # 使用最小-最大归一化公式: (xi - min) / (max - min)
    # 将数据缩放到[0,1]区间内
    return (xi - x_min) / (x_max - x_min)


def min_max_scale(traffic_matrix):
    """
    对整个数据集进行最小-最大归一化

    Args:
        traffic_matrix: 原始数据列表

    Returns:
        list: 归一化后的数据列表
    """
    # 找到数据集中的最小值和最大值
    x_min = min(traffic_matrix)
    x_max = max(traffic_matrix)
    # 对数据集中的每个值都进行归一化处理
    return [min_max_value(xi, x_min, x_max) for xi in traffic_matrix]


def process(traffic_matrix, alpha, adjust):
    """
    处理交通流量数据，使用指数加权移动平均或简单指数平滑

    Args:
        traffic_matrix: 原始交通流量数据列表
        alpha: 学习率参数，控制平滑程度
        adjust: 是否使用调整模式的标志

    Returns:
        list: 处理后的数据列表
    """
    # 先对原始数据进行归一化处理
    # 调用min_max_scale函数将数据缩放到[0,1]区间
    x = min_max_scale(traffic_matrix)
    # 初始化结果列表，用于存储平滑后的数据
    y = []

    # 计算衰减因子，alpha = 1 - learning_rate
    # 这个值决定了历史数据的衰减速度
    one_alpha = 1 - alpha
    # 获取数据长度
    d = len(x)

    # 根据adjust参数选择不同的平滑算法
    if adjust:
        # 使用调整模式计算指数加权移动平均
        # 这种方法给予近期数据更高的权重
        # 初始化分子和分母的累积值
        numerator_sum = 0
        denominator_sum = 0

        # 从最早的数据开始处理（从索引0到 d-1）
        for i in range(d):
            # 计算分子部分：数据值 * 衰减因子的幂次
            # 幂次越大（越早期的数据）权重越小
            # (d - i - 1)表示距离当前时刻的步数，越早期步数越大
            numerator_sum += x[i] * one_alpha ** (d - i - 1)
            # 计算分母部分：衰减因子的幂次之和
            denominator_sum += one_alpha ** (d - i - 1)
            # 计算加权平均值
            # 通过分子除以分母得到当前时刻的加权平均值
            yi = numerator_sum / denominator_sum
            # 将结果添加到列表中
            y.append(yi)

    else:
        # 使用简单指数平滑算法
        # 这是一种递推算法，当前值依赖于前一个平滑值
        for i in range(d):
            if i == 0:
                # 第一个值直接使用原始数据
                # 对于第一个数据点，没有历史信息可以参考
                yi = x[i]
            else:
                # 使用指数平滑公式: yt = α * xt + (1-α) * yt-1
                # 其中α是learning_rate，yt-1是上一个平滑值
                # 这个公式给当前值分配权重α，给历史值分配权重(1-α)
                yi = alpha * x[i] + one_alpha * y[i - 1]
            # 将结果添加到列表中
            y.append(yi)
    # 返回处理后的数据
    return y
