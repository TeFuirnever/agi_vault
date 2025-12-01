import numpy as np  # 导入numpy库，用于数值计算


class KMeans:
    """
    This class is used to implement the K-Means algorithm.
    算法步骤
    初始化：随机选择K个数据点作为初始簇中心
    分配步骤：将每个数据点分配到最近的簇中心
    更新步骤：重新计算每个簇的质心
    收敛判断：重复步骤2-3直到簇中心不再变化或达到最大迭代次数
    """

    def __init__(self, k: int):
        """
        This function is used to initialize the K-Means algorithm.

        Args:
            k (int): 聚类的数量，即需要将数据分为k个簇
        """
        self.k = k  # 保存聚类数量k

    def cluster(self, points: np.ndarray, centers: np.ndarray):
        """
        执行K-Means聚类算法

        Args:
            points (np.ndarray): 需要聚类的数据点集合，形状为(n, d)，n为点数，d为维度
            centers (np.ndarray): 初始聚类中心点，形状为(k, d)，k为聚类数

        Returns:
            list: 排序后的聚类中心点列表，各中心点按照指定规则排序
        """
        # 迭代优化聚类中心，迭代k次
        # 注意：这里使用k作为迭代次数，但通常K-Means使用固定迭代次数或收敛条件
        for _ in range(self.k):
            # 初始化分类字典，为每个聚类中心创建一个空列表
            # clf字典的键是聚类中心索引，值是属于该簇的所有数据点
            clf = {}
            for i in range(len(centers)):
                clf[i] = []  # 为每个聚类中心创建空列表

            # 将每个点分配给最近的聚类中心（分配步骤）
            for point in points:
                # 计算当前点到各个聚类中心的欧几里得距离
                # np.linalg.norm计算向量的范数（默认是2-范数，即欧几里得距离）
                distances = [np.linalg.norm(point - center)
                             for center in centers]
                # np.argmin返回最小距离的索引，即最近的聚类中心
                minIdx = np.argmin(distances)
                # 将当前点添加到对应的聚类簇中
                clf[minIdx].append(point)

            # 更新聚类中心为各簇的平均值（更新步骤）
            for i in range(len(centers)):
                # 计算第i个簇中所有点的平均值作为新的聚类中心
                # axis=0表示沿着第一个维度（行）求平均，即对所有点的每个维度分别求平均
                if (len(clf[i]) > 0):
                    centers[i] = np.mean(clf[i], axis=0)

        # 对结果进行排序并返回
        # 按照第一维（x坐标）和第二维（y坐标）进行排序
        # 这样可以确保输出结果的一致性
        return sorted(centers.tolist(), key=lambda x: (x[0], x[1]))

    def cluster_weighted(
            self, points: np.ndarray, centers: np.ndarray, weights: np.ndarray
    ):
        """
        带权重的K-Means聚类

        Args:
            points (np.ndarray): 需要聚类的数据点集合，形状为(n, d)，n为点数，d为维度
            centers (np.ndarray): 初始聚类中心点，形状为(k, d)，k为聚类数
            weights (np.ndarray): 权重向量，形状为(d,)，d为数据维度

        Returns:
            list: 排序后的聚类中心点列表，各中心点按照指定规则排序
        """
        # 迭代优化聚类中心，迭代k次
        for _ in range(self.k):
            # 初始化分类字典，为每个聚类中心创建一个空列表
            clf = {}
            for i in range(len(centers)):
                clf[i] = []

            # 将每个点分配给最近的聚类中心（基于加权距离）
            for point in points:
                # 计算加权欧几里得距离
                distances = []
                for center in centers:
                    # 方法1：对每个维度应用权重
                    weighted_diff = weights * (point - center)
                    distance = np.linalg.norm(weighted_diff)
                    distances.append(distance)

                minIdx = np.argmin(distances)
                clf[minIdx].append(point)

            # 更新聚类中心为各簇的平均值（更新步骤）
            for i in range(len(centers)):
                if len(clf[i]) > 0:
                    centers[i] = np.mean(clf[i], axis=0)

        # 对结果进行排序并返回
        return sorted(centers.tolist(), key=lambda x: (x[0], x[1]))


def input_format(line):
    """
    解析输入字符串为二维浮点数列表

    Args:
        line (str): 输入的一行字符串，格式如 "1.0,2.0;3.0,4.0"

    Returns:
        list: 解析后的二维浮点数列表
    """
    ans = []
    for t in line.split(";"):
        ans.append([float(d) for d in t.split(",")])
    return ans


def out_format(output):
    """
    格式化输出聚类中心点

    Args:
        output (list): 聚类中心点列表

    Returns:
        None: 直接打印输出结果
    """
    print(";".join([",".join(["%.2f" % d for d in center])
                    for center in output]))


if __name__ == "__main__":
    # # 示例向量
    # v = np.array([3, 4, 5])

    # # 1. L0范数（非零元素个数）
    # l0_norm = np.count_nonzero(v)  # 结果为3

    # # 2. L1范数（曼哈坦距离）
    # l1_norm = np.linalg.norm(v, ord=1)  # |3|+|4|+|5| = 12

    # # 3. L2范数（欧几里得距离，默认）
    # l2_norm = np.linalg.norm(v)  # √(3²+4²+5²) = √50 ≈ 7.07

    # # 4. L∞范数（切比雪夫距离）
    # linf_norm = np.linalg.norm(v, ord=np.inf)  # max(|3|,|4|,|5|) = 5

    # # 5. Lp范数（p-范数）
    # lp_norm = np.linalg.norm(v, ord=3)  # (3³+4³+5³)^(1/3)

    # 测试参数设置
    k = 2
    p = 10
    cline = '114.0,86.0;116.0,84.0'
    dline = '119.37,81.03;119.73,79.51;121.51,81.29;120.17,79.14;119.7,79.64;111.49,90.02;109.66,91.63;110.27,89.6;110.36,88.7;110.87,89.50'
    iteration = 3

    # 使用带维度权重的K-means
    # 为每个维度设置权重（示例）
    dimension_weights = [1.5, 0.5]  # 第一个维度更重要

    # 解析初始聚类中心和数据点
    centers = np.array([[float(j) for j in i.split(',')] for i in cline.split(';')])
    pointers = np.array([[float(j) for j in i.split(',')] for i in dline.split(';')])
    # 创建KMeans实例并执行聚类
    # k = int(input())
    # centers = input_format(input())
    # points = input_format(input())
    # 110.53, 89.89;120.10, 80.12
    out_format(KMeans(iteration).cluster(np.array(pointers), np.array(centers)))
