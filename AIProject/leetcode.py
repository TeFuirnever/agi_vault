def bs(a, t):
    """
    在有序数组中使用二分查找寻找目标值的索引。

    参数:
        a (list): 一个升序排列的列表。
        t (any): 要查找的目标元素。

    返回:
        int: 目标元素在列表中的索引，如果未找到则返回 -1。
    """
    l, r = 0, len(a) - 1
    while l <= r:
        m = (l + r) // 2
        if a[m] < t:
            l = m + 1
        elif a[m] > t:
            r = m - 1
        else:
            return m
    return -1


class SegT:
    """
    线段树类，支持区间更新和单点查询操作。

    属性:
        n (int): 原始数组长度。
        sz (int): 线段树大小（2的幂）。
        d (any): 默认值。
        f (function): 合并子节点值的函数，默认为 min。
        t (list): 线段树节点值。
        lz (list): 懒惰标记数组。
    """

    def __init__(self, a, d=10 ** 18, f=min):
        """
        初始化线段树。

        参数:
            a (list): 初始数组。
            d (any): 默认值。
            f (function): 合并函数。
        """
        self.n = len(a)
        self.sz = 1
        while self.sz < self.n: self.sz *= 2
        self.d, self.f = d, f
        self.t = [0] * (2 * self.sz)
        self.lz = [0] * (2 * self.sz)
        for i, v in enumerate(a): self.t[self.sz + i] = v
        for i in range(self.sz - 1, 0, -1): self.t[i] = f(self.t[2 * i], self.t[2 * i + 1])

    def push(self, i):
        """
        将懒惰标记下传给子节点。

        参数:
            i (int): 当前节点索引。
        """
        if self.lz[i]:
            for c in (2 * i, 2 * i + 1):
                self.t[c] += self.lz[i]
                self.lz[c] += self.lz[i]
            self.lz[i] = 0

    def upd(self, l, r, v, i=1, il=0, ir=None):
        """
        区间更新操作。

        参数:
            l (int): 更新区间的左端点。
            r (int): 更新区间的右端点。
            v (any): 更新的值。
            i (int): 当前节点索引。
            il (int): 当前区间的左边界。
            ir (int): 当前区间的右边界。
        """
        if ir is None: ir = self.sz
        if r <= il or ir <= l: return
        if l <= il and ir <= r:
            self.t[i] += v;
            self.lz[i] += v;
            return
        self.push(i)
        m = (il + ir) // 2
        self.upd(l, r, v, 2 * i, il, m)
        self.upd(l, r, v, 2 * i + 1, m, ir)
        self.t[i] = self.f(self.t[2 * i], self.t[2 * i + 1])

    def qry(self, p, i=1, il=0, ir=None):
        """
        查询某个位置的值。

        参数:
            p (int): 查询的位置。
            i (int): 当前节点索引。
            il (int): 当前区间的左边界。
            ir (int): 当前区间的右边界。

        返回:
            any: 该位置的值。
        """
        if ir is None: ir = self.sz
        if ir - il == 1: return self.t[i]
        self.push(i)
        m = (il + ir) // 2
        return self.qry(p, 2 * i, il, m) if p < m else self.qry(p, 2 * i + 1, m, ir)


class Fenw:
    """
    树状数组（Fenwick Tree），支持前缀和与第k小元素查询。

    属性:
        n (int): 数组大小。
        f (list): 树状数组存储结构。
    """

    def __init__(self, n):
        """
        初始化树状数组。

        参数:
            n (int): 数组大小。
        """
        self.n = n;
        self.f = [0] * (n + 1)

    def upd(self, i, d):
        """
        更新指定位置的值。

        参数:
            i (int): 要更新的位置（从0开始）。
            d (int): 增量值。
        """
        i += 1
        while i <= self.n:
            self.f[i] += d;
            i += i & -i

    def pref(self, i):
        """
        计算前缀和。

        参数:
            i (int): 前缀结束位置（从0开始）。

        返回:
            int: 前缀和。
        """
        s = 0;
        i += 1
        while i:
            s += self.f[i];
            i -= i & -i
        return s

    def kth(self, k):
        """
        查找第k小的元素索引。

        参数:
            k (int): 第k小元素（从1开始计数）。

        返回:
            int: 第k小元素的索引。
        """
        idx = 0;
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            t = idx + bit
            if t <= self.n and self.f[t] < k:
                idx = t;
                k -= self.f[t]
            bit >>= 1
        return idx


class DSU:
    """
    并查集（Disjoint Set Union）数据结构，支持路径压缩和按秩合并。

    属性:
        p (list): 父节点数组，负数表示根节点且其绝对值为集合大小。
        c (int): 当前连通分量的数量。
    """
    __slots__ = ('p', 'c')

    def __init__(self, n):
        """
        初始化并查集。

        参数:
            n (int): 元素总数。
        """
        self.p = [-1] * n
        self.c = n

    def find(self, x):
        """
        查找元素所属集合的根节点，并进行路径压缩优化。

        参数:
            x (int): 元素索引。

        返回:
            int: 根节点索引。
        """
        p = self.p
        while p[x] >= 0:
            if p[p[x]] >= 0:
                p[x] = p[p[x]]
            x = p[x]
        return x

    def unite(self, a, b):
        """
        合并两个元素所在的集合。

        参数:
            a (int): 第一个元素。
            b (int): 第二个元素。

        返回:
            bool: 如果两个元素原本不在同一集合则返回 True，否则返回 False。
        """
        p = self.p
        a = self.find(a);
        b = self.find(b)
        if a == b: return False
        if p[a] > p[b]: a, b = b, a
        p[a] += p[b];
        p[b] = a
        self.c -= 1
        return True
