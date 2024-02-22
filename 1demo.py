# n = 5
# res = 1
# for i in range(1, n+1):
#     res *= i
# print(res)
import random
import time

t_local = time.localtime()
t_UTC = time.gmtime()
print("t_local", t_local)  # 本地时间
print("t_UTC", t_UTC)  # utc时间

ctime = time.ctime()
print("ctime", ctime)  # 本地时间字符串

t_1 = time.time()
t_2 = time.perf_counter()
t_3 = time.process_time()

print(t_1)
print(t_2)
print(t_3)

res = 0
for i in range(10000):
    res += i

# time.sleep(5)
t_1_e = time.time()  # 时间戳
t_2_e = time.perf_counter()  # 随机取一个点到现在的时间，记录sleep
t_3_e = time.process_time()  # 随机取一个点到现在的时间，不记录sleep

print(t_1_e)
print(t_2_e)
print(t_3_e)

print("time方法：{:.3f}秒".format(t_1_e - t_1))
print("perf_counter方法：{:.3f}秒".format(t_2_e - t_2))
print("process_time方法：{:.3f}秒".format(t_3_e - t_3))

localtime = time.localtime()
strf_time = time.strftime("%Y-%m-%d %A %H:%M:%S", localtime)
print(strf_time)

print("------random---------")
from random import *

# seed(10)  # 随机种子
print(random())
# seed(10)  # 随机种子
print(random())
# randint(a,b)  # 产生[a，b]之间的随机整数
# randrange(a, b)  # 产生[a,b) 之间的随机整数
# randrange(a, b, step)  # 产生[a,b) 之间以step为步长的随机整数

numbers = [randint(1, 10) for i in range(10)]  # 产生1，10之间的随机整数
print(numbers)

# uniform(a,b)  # 产生[a,b] 之间的随机浮点数

print("------序列用函数---------")
choice(['win', 'lose', 'draw'])  # 从序列中随机返回一个元素
choice('python')

choices(['win', 'lose', 'draw'], k=2)  # 对序列进行k次重复采样，可设置权重
choices(['win', 'lose', 'draw'], [4, 4, 2], k=2)
shuffle(['win', 'lose', 'draw'])  # 将序列中的元素随机排列，返回打乱后的序列

sample([10, 20, 30, 40, 50], k=5)  # 从pop类型中随机取k个元素，以列表形式返回，注意此时的k要小于列表的长度

gauss(0, 1)  # 产生一个符合高斯分布的随机数 mu 中值， sigma标准差


# 例子 随机红包
def red_packet(total, num):
    for i in range(1, num):
        per = uniform(0.01, total / (num - i + 1) * 2)  # 保证每个人获得的红包期望是total/num
        total = total - per
        print("第{}位红包金额，{:.2f}元".format(i, per))
    else:
        print("第{}位红包金额，{:.2f}元".format(num, total))


red_packet(10, 5)

# 四位验证码
import string

print(string.digits)  # 获取数字
print(string.ascii_letters)  # 获取英文字符

s = string.digits + string.ascii_letters
v = sample(s, 4)
print(v)
print(''.join(v))

# collections
import collections

# 具名元组
Card = collections.namedtuple("Card", ['rank', 'suit'])
ranks = [str(n) for n in range(2, 11)] + list('JQKA')
suits = "spades diamonds clubs hearts".split()
print("ranks:", ranks)
print("suites:", suits)
cards = [Card(rank, suit) for rank in ranks for suit in suits]
print(cards)

# 洗牌
shuffle(cards)

# 抽牌
c = choice(cards)
print(c)

# 抽多张牌
ls = sample(cards, 5)
print(ls)

# 统计
s = "啊随机发哈就开发"
colors = ["red", "blue", "green", "red"]
print(collections.Counter(s))
color_count = collections.Counter(colors)
print(color_count)
print(isinstance(collections.Counter(), dict))

# 获取n个频率最高的元素和计数
print(color_count.most_common(2))
# 展开
for a in color_count.elements():
    print(a)
print(list(color_count.elements()))

a = collections.Counter(a=1, b=3)
b = collections.Counter(a=4, b=6)
print(a + b)

# 例 从一副牌中抽取10张拍， 大于10的比例是多少
cards_counter = collections.Counter(tens=16, low_tens=36)
seen = sample(list(cards_counter.elements()), 20)
print(seen)
print(seen.count("tens") / 20)

# deque 双向队列
from collections import deque

d = deque("abd")
d.append("d")
d.appendleft("0")

d.pop()
d.popleft()

# itertools库 迭代器
import itertools

# 1 排列组合迭代器
# 1.1 product 笛卡尔积

p = itertools.product('ABC', '01')
print(list(p))

for i in itertools.product('ABC', repeat=3):
    print(i)

print('-----')
# 1.2排列 permutations
for i in itertools.permutations("ABCD", 3):  # 3 是排列的长度
    print(i)

print('-----')
for i in itertools.permutations(range(3)):
    print(i)

# 1.3 组合 combinations !!注意 组合的r参数不能缺省
print('------')
for i in itertools.combinations('ABCD', 2):
    print(i)

for i in itertools.combinations(range(4), 3):
    print(i)

# 1.4 元素可重复组合
for i in itertools.combinations_with_replacement('ABC', 2):
    print(i)

for i in itertools.product('ABC', repeat=2):
    print(i)

# 2拉链

# 2.1 zip 短拉链
for i in zip('ABC', '012', 'xyz'):
    print(i)

for i in zip('ABC', '012345'):  # 长度不一致时，执行到最短对象为止
    print(i)

# 2.2 zip_longest 长拉链
for i in itertools.zip_longest('ABC', '012345'):  # 长度不一致时，执行到最长对象为止， 缺省元素用none 或者指定元素代替
    print(i)

for i in itertools.zip_longest('ABC', '012345', fillvalue="?"):
    print(i)

# 3无穷迭代器

# 3.1 count(start=0, step=1)  计数， 创建一个迭代器，从start值开始，返回均价间隔的值
# ### itertools.count(10)
# 3.2 itertools.cycle(iterable)  循环， 创建一个迭代器，返回iterable 中的所有元素，无限重复
# ### itertools.cycle('ABC')
# 3.3 repeat(obj[, times]) 重复, 创建一个迭代器，不断重复obj， 除非设定参数times， 否则将无限重复
# ###itertools.repeat(10, 3)


# 4。 其他
# 4.1 chain(iterable) 锁链， 把一组迭代对象串联起来，形成一个更大的迭代器
for ch in itertools.chain('ABC', [1, 2, 3]):
    print(ch)

# 4.2 enumerate(iterable, start=0)  枚举（python 内置）,  产生由两个元素组成的元组，结构是（index, item）,其中index从start开始，item从iterable中获取
for enu in enumerate('Python', start=1):
    print(enu)

# 4.3 group by (iterable, key=None)  分组
# 创建一个人迭代器，按照key指定的方式，返回iterable中连续的建和组
# 一般来说，要预先对数据进行排序
# key为None默认把连续重复元素分组
for key, group in itertools.groupby('AAAAABBBBBCCCCAAADD'):
    print(key, list(group))

animals = ["duck", "eagle", "rat", "giraffe", "bear", "bat", "dolphin", "shark", "lion"]
animals.sort(key=len)  # 根据长度排序
print(animals)
for key, group in itertools.groupby(animals, key=len):
    print(key, list(group))


animals.sort(key=lambda x: x[0])  # 根据首字母排序
print(animals)
for key, group in itertools.groupby(animals, key=lambda x: x[0]):
    print(key, list(group))










