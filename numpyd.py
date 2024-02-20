import numpy as np


# 求 100万个数的倒数
def compute_reciprocals(vals):
    res = []
    for value in vals:
        res.append(1 / value)
    return res


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
