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
data_frame = pd.DataFrame(np.random.randint(10, size=(3, 3)), columns=["foo", "bar", "zoo"], index=["a", "b", "c"])
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
# 索引
# 获取列
print("----")
print(data_frame["foo"])  # 等价 data_frame.foo
print("----")
print(data_frame[["bar", "foo"]])
# 获取行
print(data_frame.loc["a"])  # 绝对索引
print(data_frame.loc[["a", "c"]])
print(data_frame.iloc[0])  # 相对索引
print(data_frame.iloc[[0, 2]])
print("------")
print(data_frame.loc["a", "foo"])  # 标量
print(data_frame.iloc[0, 1])
print(data_frame.values[0, 1])
# 切片
date_range = pd.date_range(start="2019-01-01", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=date_range, columns=["A", "B", "C", "D"])
print(df)
print(df["2019-01-01":"2019-01-03"])  # 行切片
print(df.loc["2019-01-01":"2019-01-03"])
print(df.iloc[0:3])
print(df.loc[:, "A":"C"])  # 列切片
print(df.iloc[:, 0:3])
# print(df[:, "A":"C"])  # 这样不行
print(df.loc["2019-01-02":"2019-01-04", "C":"D"])  # 同时切片
print(df.iloc[1:4, 2:])  # 同时切片
print(df.loc["2019-01-01":"2019-01-03", ["A", "C"]])  # 行切片，列分散取值  df.iloc[:3, [0, 2]]
print(df.loc[["2019-01-04", "2019-01-06"], "C": "D"])  # 行分散，列分切片  df.iloc[[1, 5], 0:3]
print(df.loc[["2019-01-04", "2019-01-06"], ["B", "D"]])  # 行列都分散  df.iloc[[3, 5], [1, 3]]
# 布尔索引
print(df > 0)
print(df[df > 0])
print(df.A > 0)
print(df[df.A > 0])
print("------")
copy = df.copy()
copy["E"] = ["one", "one", "two", "three", "four", "three"]  # 给copy加一列“E”
ind = copy["E"].isin(["two", "four"])  # isin 方法
print(copy)
print(ind)
print(copy[ind])
# 赋值
print("------")
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20190101", periods=6))
print(s1)
df["E"] = s1  # 增加新列， 用列表或者Series都可以
print(df)
df.loc["2019-01-01", "A"] = 0  # 修改赋值 df.iloc[0,0] = 0
df["D"] = np.array([5] * len(df))  # 也可 df["D"] = 5
print(df)
df.index = [0, 1, 2, 3, 4, 5]
df.columns = [i for i in range(df.shape[1])]
print(df)
# --------------------------
# 数值运算及统计分析
print("---------------------------")
date_range = pd.date_range(start="2019-01-01", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=date_range, columns=["A", "B", "C", "D"])
print(df)
# 数据查看
print(df.head())  # 查看前面的行，默认5行
print(df.head(2))
print(df.tail())  # 查看后面的行
print(df.tail(3))
print(df.info())  # 总体信息

# 通用函数 与Numpy 通用
A = pd.DataFrame(np.random.randint(0, 20, size=(2, 2)), columns=list("AB"))  # 2*2
B = pd.DataFrame(np.random.randint(0, 10, size=(3, 3)), columns=list("ABC"))  # 3*3
print(A + B)  # 自动用NaN 补齐
print(A.add(B, fill_value=0))  # 也可以指定填充
#  -----------------------
# 统计相关
# 数据种类的统计
y = np.random.randint(3, size=20)
print(y)
#  现在想要统计 0， 1， 2 分别有多少个
from collections import Counter

print(Counter(y))  # 1.使用 collections库里的counter方法

y1 = pd.DataFrame(y, columns=["A"])
print(y1)
print(y1["A"].value_counts())  # 2.数据种类的统计
print(y1.sort_values(by="A"))  # 排序
print(y1.sort_values(by="A", ascending=False))  # 排序 递减
print(df)
print(df.sort_index())  # 对行进行排序
print(df.sort_index(axis=1))  # 对列进行排序
print(df.sort_index(axis=1, ascending=False))  # 对列 倒序排序
print(df.count())  # 非空个数
print(df.sum())  # 对每一列求和  df.sum(axis=1)
print(df.min())
print(df.max())
print(df.idxmin())  # 最小值的坐标
print(df.idxmax())
print(df.mean())  # 均值


# var方差  std标准差  median中位数  mode众数 df.quantile(0.75)75%分位数
# df.describe() 一网打尽
# df.corr() 相关性系数
# df.corrwith(df["A"]) 某一列相关性系数
# 自定义操作
# apply(method) 使用method 方法默认对每一列进行相应的操作
# df.apply(np.cumsum)  每一行依次累加 类似斐波那契数列计算方式  df.apply(np.cumsum, axis=1) 按列
# df.sum() 等价 df.apply(sum)
# df.apply(lambda x: x.max()-x.min())
def my_describe(x):
    return pd.Series([x.count(), x.mean()], index=["count", "mean"])


print(df.apply(my_describe))

# ---------------------------
# 处理缺失值
# 有None,字符串等，数据类型变更成object, 他比int和float更消耗资源
# 注意np.nan 是一种特殊的浮点数
pd_data_frame = pd.DataFrame(np.array([[1, np.nan, 2], [np.nan, 3, 4], [5, 6, None]]), columns=["A", "B", "C"])
print(pd_data_frame)
print(pd_data_frame.dtypes)
print(pd_data_frame.isnull())
print(pd_data_frame.notnull())


# pd_data_frame.dropna() 删除整行
# pd_data_frame.dropna(axis="columns") 按列删除
# pd_data_frame.dropna(axis="columns", how="all")  how 判断策略， 默认any , all表示必须全部为缺失才会删除
# pd_data_frame.fillna(value=5)  填充缺失值
# pd_data_frame.fillna(value=pd_data_frame.mean())  用均值替换
# pd_data_frame.fillna(value=pd_data_frame.stack().mean())  用全部数的均值替换 stack相当于展开了
# 合并数据
def make_df(cols, ind):
    # 一个简单的DataFrame
    data = {c: [str(c) + str(i) for i in ind] for c in cols}  # 这里构造了一个字典
    return pd.DataFrame(data, ind)


print(make_df("ABC", range(3)))

df_1 = make_df("AB", [1, 2])
df_2 = make_df("AB", [3, 4])
print(pd.concat([df_1, df_2]))  # 垂直合并
df_3 = make_df("AB", [1, 2])
df_4 = make_df("CD", [1, 2])
print(pd.concat([df_3, df_4], axis=1))  # 水平合并
df_5 = make_df("AB", [1, 2])
df_6 = make_df("AB", [1, 2])
print(pd.concat([df_5, df_6], ignore_index=True))  # 对index 重新编号
df_7 = make_df("ABC", [1, 2])
df_8 = make_df("BCD", [3, 4])
print(pd.concat([df_7, df_8], sort=False))  # 列重叠
df_9 = make_df("AB", [1, 2])
df_10 = make_df("BC", [1, 2])
print(pd.concat([df_9, df_10]))
print(pd.merge(df_9, df_10))  # 对齐合并默认交集   print(pd.merge(df_9, df_10, how="outer")) outer 并集
# ---------------------------
# 分组和数据透视表
df = pd.DataFrame({"key": list("ABCABC"),
                   "data1": range(6),
                   "data2": np.random.randint(0, 10, size=6)})
print(df)
print(df.groupby("key"))
print(df.groupby("key").sum())
print(df.groupby("key").mean())
for i in df.groupby("key"):
    print(str(i))
print(df.groupby("key")["data2"].sum())  # 按列取值

# ---------------------------
# 其他
