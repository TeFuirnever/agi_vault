def find_cycle_edge():
    """
    查找图中形成环路的最后一条边
    
    该函数读取图的边信息，构建图结构，使用深度优先搜索(DFS)检测环路，
    并找出形成环路的最后一条边并输出。
    
    Returns:
        None: 直接打印结果，无返回值
    """
    # 读取边数
    # 从标准输入读取图中的边数n
    n = int(input().strip())

    # 存储所有边的列表
    # 用于保存输入的所有边，按输入顺序存储
    edges = []

    # 图的邻接表表示
    # 使用字典存储图结构，键为节点，值为邻接节点列表
    graph = {}

    # 读取所有边并构建图
    # 循环n次读取所有边信息
    for i in range(n):
        # 读取一条边的两个端点u和v
        u, v = map(int, input().strip().split())
        # 将边添加到边列表中
        edges.append((u, v))

        # 初始化节点的邻接表
        # 如果节点u不在图中，创建其邻接表
        if u not in graph:
            graph[u] = []
        # 如果节点v不在图中，创建其邻接表
        if v not in graph:
            graph[v] = []

        # 添加无向边，同时记录边的索引
        # 在无向图中，每条边需要在两个方向都添加
        # 同时记录边在输入中的索引i，用于后续识别具体是哪条边
        graph[u].append((v, i))
        graph[v].append((u, i))

    # DFS访问标记数组
    # 用于标记节点是否已被访问，避免重复访问
    visited = [False] * (n + 2)

    # 父节点记录数组
    # 记录每个节点在DFS过程中的父节点，用于避免将父节点误判为环路
    parent = [-1] * (n + 2)

    # 存储环路中边的索引集合
    # 使用集合存储构成环路的边的索引
    edge_index_in_cycle = set()

    # 环路入口节点
    # 记录检测到的环路入口节点
    entry_node = -1

    def dfs(u, p):
        """
        深度优先搜索检测环路

        Args:
            u: 当前访问的节点
            p: 父节点

        Returns:
            int: 如果找到环路返回入口节点，否则返回-1
        """
        # 声明使用外层函数的entry_node变量
        nonlocal entry_node

        # 标记当前节点已访问
        # 将当前节点u标记为已访问，避免重复访问
        visited[u] = True

        # 遍历当前节点的所有邻接节点
        # 遍历节点u的所有邻接节点及其对应的边索引
        for v, idx in graph[u]:
            # 跳过父节点
            # 避免将父节点误判为环路（因为是无向图）
            if v == p:
                continue

            # 如果邻接节点已访问，说明找到环路
            # 在DFS过程中遇到已访问的节点，说明存在环路
            if visited[v]:
                # 记录环路入口节点
                entry_node = v
                # 将构成环路的边索引添加到集合中
                edge_index_in_cycle.add(idx)
                # 返回当前节点，表示找到了环路
                return u
            else:
                # 设置父节点并递归访问
                # 设置节点v的父节点为u
                parent[v] = u
                # 递归访问节点v
                res = dfs(v, u)

                # 如果子节点找到了环路
                if res != -1:
                    # 将当前边添加到环路边集合中
                    edge_index_in_cycle.add(idx)

                    # 如果回到起始节点，结束搜索
                    # 当返回的节点等于当前节点时，说明环路已完整找到
                    if res == u:
                        return -1
                    # 否则继续向上传递环路信息
                    return res

        # 如果没有找到环路，返回-1
        return -1

    # 对所有节点进行DFS搜索
    # 遍历所有可能的节点，确保能检测到所有连通分量中的环路
    for i in range(1, n + 1):
        # 如果节点未被访问过，则从该节点开始DFS
        if not visited[i]:
            # 如果DFS找到了环路，则停止搜索
            if dfs(i, -1) != -1:
                break

    # 从后往前查找第一条在环路中的边并输出
    # 按输入顺序从后往前查找，找到第一条构成环路的边
    for i in range(n - 1, -1, -1):
        # 如果边索引在环路边集合中
        if i in edge_index_in_cycle:
            # 获取边的两个端点
            u, v = edges[i]
            # 输出这条边
            print(f"{u} {v}")
            # 结束函数
            return
