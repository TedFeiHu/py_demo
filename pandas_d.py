#  pandas 特点: 基于NUMPY库， 提供侧重于数据分析的高级工具
# Numpy在向量化的数值计算中表现优异
# 但是在处理更灵活复杂的数据任务：如为数据添加标签，处理缺失值，分组，透视表等方面显得力不从心
# 基于Numpy构建的Pandas库，提供了使得数据分析变得更快更简单的高级数据结构和操作工具

import pandas as pd
import numpy as np

# 对象创建 （series， DataFrame）

# Pandas Series 对象  series 是带标签数据的一维数组
# 通用结构：pd.Series(data, index=, dtype=)
# data: 数据，可以是列表，字典，或者numpy数组
# index 缺省，默认为整数序列
data = pd.Series([1.5, 3, 4.5, 6])
print(data)
# 增加index
data = pd.Series([1.5, 3, 4.5, 6], index=['a', 'b', 'c', 'd'])
print(data)
# 增加数据类型, 缺省则自动判断
# 支持多种类型, object
data = pd.Series([1, 3, '4', 6], index=['a', 'b', 'c', 'd'])
print(data)
# 也可以强制转换
data = pd.Series([1, 3, '4', 6], index=['a', 'b', 'c', 'd'], dtype=float)
print(data)

# 用一维的numpy数组来创建
data = pd.Series(np.arange(10))
print(data)

# 用字典创建
pp_dict = {"beijing": 2154, "shanghai": 2424, "shenzhen": 1303, "hangzhou": 981}
data = pd.Series(pp_dict)
print(data)
# 用字典的情况下，如果指定index, 则会到键中寻找，找不到则设为NaN
data = pd.Series(pp_dict, index=["beijing", "hangzhou", "c", "d"])
print(data)

# data 为标量的情况
data = pd.Series(5, index=[100, 200, 300])
print(data)
# ------------------------
# DataFrame 对象 是带标签数据的的多维数组
# pd.DataFrame(data, index=, columns=)

# 通过series 对象创建
pop_series = pd.Series(pp_dict)
data_frame = pd.DataFrame(pop_series)  # #创建对象
print(data_frame)
data_frame = pd.DataFrame(pop_series, columns=["pop"])
print(data_frame)

gdp_dict = {"beijing": 30000, "shanghai": 40000, "shenzhen": 50000, "hangzhou": 60000}
gdp_series = pd.Series(gdp_dict)
data_frame = pd.DataFrame({"pop": pop_series, "gdp": gdp_series})
print(data_frame)
data_frame = pd.DataFrame({"pop": pop_series, "gdp": gdp_series, "country": "China"})  # 自动补齐
print(data_frame)

# 通过字典列表对象创建
data = [{"a": i, "b": i * 2} for i in range(3)]
data_frame = pd.DataFrame(data)  # 字典的索引作为index 字典的键作为colums
print(data_frame)
data = [{"a": 1, "b": 1}, {"b": 3, "c": 4}]
data_frame = pd.DataFrame(data)  # 不存在的，默认为NaN
print(data_frame)

# 通过Numpy二维数组创建
data_frame = pd.DataFrame(np.random.randint(10, size=(3, 2)), columns=["foo", "bar"], index=["a", "b", "c"])
print(data_frame)

# ------------------------
# DataFrame 的性质
# 属性
print(data_frame.values)  # 返回Numpy 数组表示的数据
print(data_frame.columns)  # 返回列索引
print(data_frame.index)  # 返回行索引
print(data_frame.shape)  # 返回值的形状
print(data_frame.size)  # 元素数量
print(data_frame.dtypes)  # 返回每列的数据类型
# 取值
print(data_frame["foo"])
print(data_frame[["bar", "foo"]])


