import matplotlib.pyplot as plt
import numpy as np

# 设置风格
print(plt.style.available[:5])
plt.style.use("Solarize_Light2")

# 基础用法
# x = [1, 2, 3, 4]
# y = [1, 4, 9, 16]
# plt.plot(x, y)
# plt.ylabel("squares")  # y坐标轴设置标签
# plt.show()

# with plt.style.context("seaborn-white"):
#     # 临时设置风格
#     plt.plot(x, y)
#     plt.show()

# 保存图像
# plt.savefig("my_f.png")

# 折线图
# x = np.linspace(0, 2 * np.pi, 100)  # 0到2PI 之间的100个等差数列
# plt.plot(x, np.sin(x))  # y=sin(x)
# plt.plot(x, np.cos(x))  # y=cos(x)
#
# plt.plot(x, np.cos(x - np.pi), c="r")  # color简写c 设置颜色 blue g r yellow pink
# plt.plot(x, np.cos(x - np.pi/4), ls="--")  # linestyle 简写 ls 设置线条风格 solid dashed dashdot dotted - -- -. :
# plt.plot(x, np.cos(x - np.pi/6), lw=2, c="pink")  # linewidth 设置线宽
# plt.plot(x, np.cos(x - np.pi/8), marker="*", markersize=3)  # 数据点样式 * + o圆 s方
# 颜色风格设置简写 linestyle: "g-" "c--" "k-." "r:" / "g*-" "b+--" "ko-." "rs:"  k黑色

# 调整坐标轴
# plt.plot(x, np.sin(x))
# plt.xlim(-1, 7)  # 坐标范围
# plt.ylim(-1.5, 1.5)
# plt.axis([-2, 8, -2, 2]) #  范围
# plt.axis("tight")  # 风格 紧凑tight  扁平equal
#
# x = np.logspace(0, 5, 100)  # 对数
# plt.plot(x, np.log(x))
# plt.xscale("log")  # x轴为对数坐标
#
# plt.xticks(np.arange(0, 12, step=1), fontsize=15)  # 坐标刻度
# plt.yticks(np.arange(0, 100, step=10))
# plt.tick_params(axis="both", labelsize=15)  # 刻度样式
#
# plt.title("图的名字", fontsize=20)  # 图形标签
# plt.xlabel("x")
# plt.ylabel("sin(x)")
#
# plt.plot(x, np.sin(x), "b--", label="sin")  # label 图例
# plt.legend(loc="upper center", frameon=True, fontsize=5)  # 添加图例
#
# plt.text(3.5, 0.5, "y=sin(x)")  # 添加文字
#
# plt.annotate("local min", xy=(np.pi * 3 / 4, -1), xytext=(4.5, 0))  # 箭头


# 散点图
# plt.scatter(x, np.sin(x), marker="+", s=30, c="r")
#
# plt.scatter(x, y=np.sin(x), c=np.sin(x), cmap="Blues")  # 渐变色， 随着y值渐变  s也可以是变化的
# plt.colorbar()
# plt.scatter(x, y=np.sin(x), c=np.sin(x), cmap="Blues", alpha=0.3)  # 透明度


# 例 随机漫步
# from random import choice
#
#
# class RandomWalk:
#     """"一个产生随机漫步的类"""
#
#     def __init__(self, num_points=5000):
#         self.num_points = num_points
#         self.x_values = [0]
#         self.y_values = [0]
#
#     def fill_walk(self):
#         while len(self.y_values) < self.num_points:
#             x_direction = choice([1, -1])
#             x_distance = choice([0, 1, 2, 3, 4])
#             x_step = x_distance * x_direction
#
#             y_direction = choice([1, -1])
#             y_distance = choice([0, 1, 2, 3, 4])
#             y_step = y_distance * y_direction
#
#             if x_step == 0 or y_step == 0:
#                 continue
#             next_x = self.x_values[-1] + x_step
#             next_y = self.y_values[-1] + y_step
#             self.x_values.append(next_x)
#             self.y_values.append(next_y)
#
#
# rw = RandomWalk(10000)
# rw.fill_walk()
# l = list(range(rw.num_points))
# plt.figure(figsize=(12, 6))  # 画布的大小
# plt.scatter(rw.x_values, rw.y_values, c=l, cmap="inferno", s=1)
# plt.colorbar()
# plt.scatter(0, 0, c="g", s=100)
# plt.scatter(rw.x_values[-1], rw.y_values[-1], c="r", s=100)
# plt.xticks([])
# plt.yticks([])


# 柱形图
# x = np.arange(1, 6)
# plt.bar(x, 2 * x, align="center", width=0.5, alpha=0.5, color='y', edgecolor='r')  # align 文字再柱子的中间
# plt.xticks(x, ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.tick_params(axis="both", labelsize=5)
#
# 累加柱形图
# plt.bar(x, y1)
# plt.bar(x, y2, bottom=y1)  # 累加柱形图，二组数据的bottom是一组的y值
#
# 并列柱形图
# plt.bar(x, y1， width=0.3)
# plt.bar(x+0.3, y2, width=0.3)
#
# 横向柱形图
# plt.barh()  # 柱宽 用height

# 多子图
# plt.subplots_adjust(hspace=0.5, wspace=0.5)  # 多图之间的间隔
# plt.subplot(211)  # 21 两行一列 第一个图
# """第一个图的绘制"""
# plt.subplot(212)  # 21 两行一列 第二个图
# """第二个图的绘制"""
#
# 不规则子图
# grid_spec = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)  # 两行三列的网格
# plt.subplot(grid_spec[0, 0])  # 占据00 网格的位置制图
# plt.plot(x, np.sin(x))
# plt.subplot(grid_spec[0, 1:])  # 占据 01 02 网格制图
# plt.plot(x, np.sin(x), c='r')
# plt.subplot(grid_spec[1, :])  # 占据 10 11 12 网格制图
# plt.plot(x, np.sin(x), c='g')

# 直方图
mu, sigma = 100, 15  # 均值为100， 标准差为15
x = mu + sigma * np.random.randn(10000)
# 普通频次直方图
plt.hist(x, bins=50, facecolor='g', alpha=0.5)


plt.show()
