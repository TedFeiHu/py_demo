from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = sns.load_dataset("iris")

print(iris.shape)
print(iris.head())
print(iris.species.value_counts())

# sns.pairplot(iris, hue='species')
# plt.show()

# ---------------数据准备
# 标签清洗
iris_simple = iris.drop(["sepal_length", "sepal_width"], axis=1)  # 丢弃两个字段
# print(iris_simple.head())
# 标签编码
# 将species标签的内容 映射成数值类型0，1，2
encoder = LabelEncoder()
iris_simple.species = encoder.fit_transform(iris_simple.species)  # iris_simple.species 等价 iris_simple["species"]
# 数据集的标准化(本数据集特征较为接近，实际处理过程中未标准化)   获得一个符合正态分布的数据集
standard_scaler = StandardScaler()
_iris_s = standard_scaler.fit_transform(iris_simple[['petal_length', 'petal_width']])
# 构建验证集和测试集（暂时不考虑验证集）
train_set, test_set = train_test_split(iris_simple, test_size=0.2)  # test_size 20%的数据做 测试集
iris_x_train = train_set[["petal_length", "petal_width"]]  # 这里是构建了一个新的DataFrame对象
iris_y_train = train_set["species"].copy()  # [] 这里是视图，所以需要copy
iris_x_test = test_set[["petal_length", "petal_width"]]
iris_y_test = test_set["species"].copy()

# 这里有个默认共识， 设有样本数据集D={D1,D2,D3...Dn} ，对应样本数据的特征属性集为X={X1,X2,X3...Xn} ,类变量为Y={Y1,Y2...Ym} ，即 D可以分为Ym类别。
# ----------------k近领算法
# 1 基本思想，与待预测点最近的训练数据集中的k个邻居， 把k个近邻中最常见的类别预测为待预测点的类别
# 2 sklearn 实现
from sklearn.neighbors import KNeighborsClassifier

klf = KNeighborsClassifier()  # 构建分类器对象
# print(vars(klf))
klf.fit(iris_x_train, iris_y_train)  # 训练
res = klf.predict(iris_x_test)  # 使用测试集 预测测试
# print(res)
# print(iris_y_test.values)
# transform = encoder.inverse_transform(res)  # 编码结果 在转成标签
# print(transform)
accuracy = klf.score(iris_x_test, iris_y_test)  # 评分
print("正确率：{:.0%}".format(accuracy))
# out = iris_x_test.copy()
# out["y"] = iris_y_test
# out["pre"] = res
# out.to_csv('./csv/iris_kn_predict.csv')  # 保存结果

# --------------朴素贝叶斯算法
# 1. 基本思想， 当 x=(x1, x2) 发生时，哪一个yk发生的概率最大
from sklearn.naive_bayes import GaussianNB

clk = GaussianNB()
# print(vars(clk))
clk.fit(iris_x_train, iris_y_train)  # 训练
clk.predict(iris_x_test)  # 预测
accuracy = clk.score(iris_x_test, iris_y_test)  # 评估
print("正确率：{:.0%}".format(accuracy))

# ---------------决策树算法
# 1. 基本思想， CART算法：每次通过一个特征，将数据尽可能的分为纯净的两类，递归的分下去
# from sklearn.tree import DecisionTreeClassifier

# ----------------逻辑回归算法
# 1. 基本思想，
#  一种解释：
#   训练：通过一个映射方式，将特征x=(x1,x2) 映射成P(y=ck),求使得所有概率之积最大化的映射方式里的参数
#   预测：计算P(y=ck) 取概率最大的那个类别，作为预测对象的分类

#  ----------------支持向量机算法

# ----------------集成方法
# --------------随机森林
# --------------Adaboost
# --------------梯度提升树GBDT

# -------------大杀器
# xgboost
# lightgbm
# stacking
# 神经网络

