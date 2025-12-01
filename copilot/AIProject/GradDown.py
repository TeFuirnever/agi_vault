from openpyxl.styles.builtins import output


def cal_grad(x, y):
    """
    计算函数在点(x,y)处的梯度

    Args:
        x: 函数自变量x的值
        y: 函数自变量y的值

    Returns:
        tuple: 包含x方向和y方向偏导数的元组
    """
    # 计算x方向的偏导数
    # 函数形式为: f(x,y) = x*y^3 + 2*x^2*y + x*y + y^2 + x + y
    x_ = 3 * x * (y ** 2) + 4 * x * y + y
    # 计算y方向的偏导数
    y_ = 3 * (x ** 2) * y + 2 * (x ** 2) + x + 2 * y + 1
    # 返回两个方向上的梯度值
    return x_, y_


def get_grad_down(gx, gy, wx, wy, A):
    """
    根据梯度和学习率更新模型权重

    Args:
        gx: x方向的梯度值
        gy: y方向的梯度值
        wx: 当前x方向的权重
        wy: 当前y方向的权重
        A: 学习率参数

    Returns:
        tuple: 更新后的x和y方向权重
    """
    # 使用梯度下降公式更新权重: w = w - learning_rate * gradient
    # 更新x方向的权重
    wx_ = wx - A * gx
    # 更新y方向的权重
    wy_ = wy - A * gy
    # 返回更新后的权重
    return wx_, wy_


def process(init_value, async_order, learning_rate):
    """
    执行异步梯度下降算法

    异步梯度下降主流程：
        首先为每个工作节点初始化权重值，存储在grad_d字典中
        然后按照给定的工作节点顺序依次处理
        每个工作节点使用自己存储的权重值计算梯度
        使用计算出的梯度更新全局权重
        将更新后的全局权重存储回当前工作节点
        最终输出优化后的权重值

    Args:
        init_value: 初始权重值[x, y]
        async_order: 异步工作节点顺序列表
        learning_rate: 学习率参数

    Returns:
        list: 最终的权重值[x, y]
    """
    # 初始化各工作节点的梯度值，为每个工作节点分配初始权重值
    # 创建一个字典，键为工作节点标识，值为初始权重值
    grad_d = {}
    for w in async_order:
        grad_d[w] = init_value

    # 获取初始权重值，分别赋给wx和wy变量
    wx, wy = init_value[0], init_value[1]

    # 按顺序处理各工作节点的梯度更新
    # 遍历异步工作节点顺序列表
    for _, worker in enumerate(async_order):
        # 获取当前工作节点存储的权重值
        # 从grad_d字典中取出当前工作节点的权重值
        x_0, y_0 = grad_d[worker][0], grad_d[worker][1]

        # 计算当前点的梯度
        # 调用cal_grad函数计算在(x_0, y_0)点的梯度
        gx, gy = cal_grad(x_0, y_0)

        # 根据梯度和学习率更新全局权重
        # 调用get_grad_down函数更新全局权重wx和wy
        wx, wy = get_grad_down(gx, gy, wx, wy, learning_rate)

        # 将更新后的全局权重存储到当前工作节点中
        # 更新当前工作节点存储的权重值为最新的全局权重
        grad_d[worker] = [wx, wy]

    # 构造输出结果
    # 将最终的权重值放入output列表中
    output = [wx, wy]

    # 打印结果，保留三位小数
    # 格式化输出最终权重值，保留3位小数
    print(f"{output[0]:.3f} {output[1]:.3f}")

    # 返回最终的权重值
    return output


if __name__ == "__main__":
    process()
