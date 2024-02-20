import numpy as np


# 求 100万个数的倒数
def compute_reciprocals(vals):
    res = []
    for value in vals:
        res.append(1 / value)
    return res


x = [0, 1, 2]
x2 = np.array([0, 1, 2])
print(x, x2)
print(type(x), type(x2))

values = compute_reciprocals(list(range(1, 10)))
print(values)

values = np.arange(1, 10)
print(1 / values)

#  创建数组
x = np.array([1, 2, 3, 4, 5])
print(type(x))
print(x.shape)

f = np.array([1, 2, 3, 4, 5], dtype="float32")
print(type(f))
print(type(f[0]))

# 二维数组
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print(x.shape)
# 创建一个长度为5的数组，但值都为0
x = np.zeros(5, dtype="int")
print(x)
# 创建一个2*4，值为1的浮点类型数组
x = np.ones((2, 4), dtype=float)
print(x)
# 创建一个3*5， 值为8.8的数组
x = np.full((3, 5), 8.8)
print(x)
# 创建一个3*3的单位矩阵
x = np.eye(3)
print(x)
# 创建一个线性序列数组，从1开始到15结束，步长为2
x = np.arange(1, 15, 2)
print(x)
# 创建一个4个元素的数组，这4个元素均匀的分配到0-1 等差数列
x = np.linspace(0, 1, 4)
print(x)
# 创建一个10个元素的数组，形成1-10^9的等比数列
x = np.logspace(0, 9, 10)
print(x)
# 创建一个3*3的， 在0-1之间均匀分布的随机数构成的数组
x = np.random.random((3, 3))
print(x)
# 创建一个3*3，均值为0，标准差为1的 随机数数组
x = np.random.normal(0, 1, (3, 3))
print(x)
# 创建一个3*3，在【0，10）之间随机整数构成的数组
x = np.random.randint(0, 10, (3, 3))
print(x)
# 随机重排
x = np.array([10, 20, 30, 40])
x = np.random.permutation(x)  # 产生新的数组
print(x)
np.random.shuffle(x)  # 修改原数组
print(x)

#  随机采样
#  按指定形式采样
x = np.arange(10, 25, dtype=float)
choice = np.random.choice(x, (4, 3))
#  按概率采样
choice = np.random.choice(x, (3, 4), p=x / np.sum(x))

#  Numpy的性质
print(choice.shape)  # 形状
print(choice.ndim)  # 维度
print(choice.size)  # 大小
print(choice.dtype)  # 类型

# 索引
# x[0][0] x[0,0]

# 数组的切片  !!!切片是视图，而不是副本
# 一维
x1 = np.arange(10)
print(x1[:3])  # 取前三个元素
print(x1[3:])  # 第三个元素到最后
print(x1[::-1])  # 倒着取

# 二维
print(choice)
print(choice[:2, :3])  # 前两行，三列
print(choice[:2, 0:3:2])  # 两行三列，隔一列
print(choice[::-1, ::-1])  # 倒转

print(choice[1, :])  # 第一行
print(choice[1])  # 第一行
print(choice[:, 2])  # 第二列

# 希望获取副本
choice[:, 2].copy()

# 数组的变形
x2 = np.random.randint(0, 10, (12,))
print(x1.shape, x2.shape)
print(x2.reshape(3, 4))  # 一维 转3*4  结构要匹配 # reshape 返回的是视图
print(x2.reshape(1, x2.shape[0]))  # 转1*12 x2.shape[0]代表了数量(12,)，也可以x2.size
print(x2[np.newaxis, :])  # 转1*12  np.newaxis 新增维度
print(x2.reshape(x2.size, 1))  # 转12*1
print(x2[:, np.newaxis])
print(choice.flatten())  # 多维转一维 flatten 返回的也是副本
print(choice.ravel())  # 多维转一维 ravel 返回的是视图
print(choice.reshape(-1))  # reshape 视图

# 数组的拼接
x1 = np.array([[1, 2, 3],
               [11, 22, 33]])
x2 = np.array([[7, 8, 9],
               [77, 88, 99]])
# 水平拼接 --非视图
x3 = np.hstack([x1, x2])
x3 = np.c_[x1, x2]
# 垂直拼接
x3 = np.vstack([x1, x2])
x3 = np.r_[x1, x2]

# 数组的分裂
# split
x1 = np.arange(10)
x2, x3, x4 = np.split(x1, [2, 7])
print(x2, x3, x4)
# hsplit
xy = np.arange(1, 26).reshape(5, 5)
r, m, l = np.hsplit(xy, [2, 4])  # r:0-1列  m:2-3列 l:第4列
print(xy)
print(r)
print(m)
print(l)
r, m, l = np.vsplit(xy, [2, 4])  # r:0-1行  m:2-3行 l:第4行
print(xy)
print(r)
print(m)
print(l)

# 四大运算
# 向量化运算
# 加减乘除 取反，平方， 整数商，求余数
x1 = np.arange(1, 6)
print(x1 + 5)
print(x1 - 5)
print(x1 * 5)
print(x1 / 5)
print(-x1)
print(x1 ** 2)
print(x1 // 2)
print(x1 % 2)
# 绝对值，三角函数， 指数， 对数
abs(x1)
np.abs(x1)
#
theta = np.linspace(0, np.pi, 3)
print("sin()", np.sin(theta))
print("cos()", np.cos(theta))
print("tan()", np.tan(theta))
x = [1, 0, -1]
print("arcsin()", np.arcsin(x))  # 反三角函数
print("arccon()", np.arccos(x))
print("arctan()", np.arctan(x))
#
print(np.exp(x))
#
x = [1, 2, 10]
print(np.log(x))
print(np.log2(x))
print(np.log10(x))

# 两个向量的运算
x1 = np.arange(1, 6)
x2 = np.arange(6, 11)
print(x1, "---", x2)
print(x1 + x2)
print(x1 - x2)
print(x1 * x2)
print(x1 / x2)
# 矩阵运算
x = np.arange(9).reshape(3, 3)
# 矩阵转置
print(x.T)
# 矩阵乘法
x = np.array([[1, 0], [1, 1]])
y = np.array([[0, 1], [1, 1]])
print(x.dot(y))
print(np.dot(y, x))

# 广播运算
# 如果两个数组再形状上不匹配， 那么数组会沿着维度为1的维度进行扩展一匹配另一个数组
x1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
x2 = np.array([[1, 1, 3]])
# x2 = np.array([[1, 1], [2, 2]]) 不可以。只能在维度为1的方向上广播
x1 = np.array([[1], [2], [3]])
x2 = np.array([[1, 2, 3]])
print(x1 + x2)

# 比较运算，掩码
ten_ten = np.random.randint(100, size=(3, 3))
print(ten_ten)
print(ten_ten > 50)
print(np.sum(ten_ten > 50))
print(np.all(ten_ten > 50))
print(np.all(ten_ten > 50, axis=1))  # 按行判断  axis=0 按列判断
print(np.any(ten_ten > 50))
# 掩码
print(ten_ten[ten_ten > 50])
# 一维 花式索引  结果的形状 与索引的形状一致
r = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
ind = [0, 1]
print(r[ind])
ind = np.array([[0, 5], [1, 8]])
print(r[ind])
# 二维
r = np.arange(12).reshape(3, 4)
row = np.array([0, 1, 2])
col = [1, 3, 0]
print(r[row, col])
# row[:, np.newaxis] 行转列 [0][1][2]
print(r[row[:, np.newaxis], col])  # 广播机制

# 通用函数
i = np.random.randint(10, 50, size=10)
print(i)
print(np.sort(i))  # 排序 产生新的数组  i.sort()  # 原数组排序
print(np.argsort(i))  # 排序前的位置索引
print(np.max(i))
print(np.min(i))
print(np.argmax(i))
print(np.argmin(i))
print(np.sum(i))  # i.sum()
# print(np.sum(x, axis=1)) 多维情况下，按行求和
# print(np.sum(x, axis=0)) 按列求和
print(np.prod(i))  # 求积 i.prod()
print(np.median(i))  # 中位数
print(np.mean(i))  # 均值 i.mean()
print(np.var(i))  # 方差 i.var()
print(np.std(i))  # 标准差 i.std()
