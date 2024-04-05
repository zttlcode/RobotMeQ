import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from decimal import Decimal


# 定义fuzzy set 7个μ的分段函数
def membership(x, w1, w2, w3):  # x是x*t，w是幅度也是x轴刻度
    if x <= w1:
        y = 0
    if x > w1 and x <= w2:
        y = (x - w1) / (w2 - w1)
    if x > w2 and x <= w3:
        y = (w3 - x) / (w3 - w2)
    if x > w3:
        y = 0

    if w1 == w2:
        if x <= w2:
            y = 1
        if x > w2 and x <= w3:
            y = (w3 - x) / (w3 - w2)
        if x > w3:
            y = 0

    if w2 == w3:
        if x <= w1:
            y = 0
        if x > w1 and x <= w2:
            y = (x - w1) / (w2 - w1)
        if x > w2:
            y = 1

    return y  # 返回μ值


# 以论文part2的函数代入为例，试验发现函数代入与我直接用，结果是一样的
# x3 = Decimal('-0.04')
w = Decimal('0.01')

x_list = []
y_list = []
'''
for i in range(80):
    y1 = membership(x3, 0, w, 2 * w)  # PS
    y2 = membership(x3, w, 2 * w, 3 * w)  # PM
    y3 = membership(x3, 2 * w, 3 * w, 3 * w)  # PL
    y4 = membership(x3, -2 * w, -w, 0)  # NS
    y5 = membership(x3, -3 * w, -2 * w, -w)  # NM
    y6 = membership(x3, -3 * w, -3 * w, -2 * w)  # NL
    y7 = membership(x3, -w, 0, w)  # AZ

    ed1 = 0
    y_div_1 = y1 + y2 + y3 + y7
    if y_div_1 != 0:
        ed1 = (Decimal('-0.1') * y1 - Decimal('0.2') * y2 - Decimal('0.4') * y3) / y_div_1

    ed2 = 0
    y_div_2 = y4 + y5 + y6 + y7
    if y_div_2 != 0:
        ed2 = (Decimal('0.1') * y4 + Decimal('0.2') * y5 + Decimal('0.4') * y6) / y_div_2

    ed_0 = 0
    y_div_0 = y1 + y2 + y3 + y4 + y5 + y6 + y7
    if y_div_0 != 0:
        ed_0 = (Decimal('0.1') * y1 + Decimal('0.4') * y2 - Decimal('0.2') * y3 - Decimal('0.1') * y4 - Decimal(
            '0.4') * y5 + Decimal('0.2') * y6 + Decimal('0') * y7) / y_div_0
    x3 += Decimal('0.001')
    if x3 == Decimal('0.02'):
        print(x3)
    x_list.append(round(x3, 3))
    y_list.append(round(ed_0, 3))

plt.plot(np.array(x_list), np.array(y_list))
plt.show()

'''

import numpy as np
import matplotlib.pyplot as plt

# 定义模糊理论的函数
def meb(x, w1, w2, w3):  # x是x*t，w是幅度也是x轴刻度
    if x <= w1:
        y = 0
    if x > w1 and x <= w2:
        y = (x - w1) / (w2 - w1)
    if x > w2 and x <= w3:
        y = (w3 - x) / (w3 - w2)
    if x > w3:
        y = 0

    if w1 == w2:
        if x <= w2:
            y = 1
        if x > w2 and x <= w3:
            y = (w3 - x) / (w3 - w2)
        if x > w3:
            y = 0

    if w2 == w3:
        if x <= w1:
            y = 0
        if x > w1 and x <= w2:
            y = (x - w1) / (w2 - w1)
        if x > w2:
            y = 1

    return y  # 返回μ值


# 自己生成个600个时刻的模拟价格序列p
n = 600
p = np.zeros(n)
p[0] = 10
p[1:3] = p[0]
a1 = np.zeros(n)
a2 = np.zeros(n)
a1[0:80] = 0
a1[80:200] = 0.2
a1[200:400] = 0
a1[400:500] = 0.4
a2[0:60] = 0
a2[60:150] = 0.15
a2[150:300] = 0.2
a2[300:450] = 0
a2[450:550] = 0.2
c = 0.01

for i in range(3, n):
    ma3 = np.sum(p[i-3:i])/3
    x3 = np.log(p[i-1]/ma3)
    y1 = meb(x3, 0, c, 2*c)
    y2 = meb(x3, c, 2*c, 3*c)
    y3 = meb(x3, 2*c, 3*c, 3*c)
    y4 = meb(x3, -2*c, -c, 0)
    y5 = meb(x3, -3*c, -2*c, -c)
    y6 = meb(x3, -3*c, -3*c, -2*c)
    y7 = meb(x3, -c, 0, c)
    y = y1 + y2 + y3 + y7
    ed1 = 0
    if y != 0:
        ed1 = (-0.1*y1 - 0.2*y2 - 0.4*y3) / y
    y = y4 + y5 + y6 + y7
    ed2 = 0
    if y != 0:
        ed2 = (0.1*y4 + 0.2*y5 + 0.4*y6) / y
    p[i] = np.exp(np.log(p[i-1]) + a1[i]*ed1 + a2[i]*ed2 + np.random.normal(0, 0.02))

plt.figure(1)
plt.plot(p)


n1 = 1
n2 = n
ma1 = 3
lmd = 0.9
P = np.eye(2)*10  # 声明单位矩阵
aa = np.zeros((2, 1, n))  # 2个 1*n的向量，用来记录两个参数在n个时刻的值
r = np.zeros(n)
error = np.zeros(n)

for k in range(n1+ma1):
    aa[:, :, k] = np.array([[0], [0]])

for k in range(n1, n2):
    r[k] = np.log(p[k]/p[k-1])

for k in range(n1+ma1, n2-1):
    pa = np.sum(p[k-ma1:k])/ma1
    x3 = np.log(p[k]/pa)
    y1 = meb(x3, 0, c, 2*c)
    y2 = meb(x3, c, 2*c, 3*c)
    y3 = meb(x3, 2*c, 3*c, 3*c)
    y4 = meb(x3, -2*c, -c, 0)
    y5 = meb(x3, -3*c, -2*c, -c)
    y6 = meb(x3, -3*c, -3*c, -2*c)
    y7 = meb(x3, -c, 0, c)
    y = y1 + y2 + y3 + y7
    ed1 = 0
    if y != 0:
        ed1 = (-0.1*y1 - 0.2*y2 - 0.4*y3) / y
    y = y4 + y5 + y6 + y7
    ed2 = 0
    if y != 0:
        ed2 = (0.1*y4 + 0.2*y5 + 0.4*y6) / y

    x = np.array([[ed1], [ed2]])

    # 这行代码的目的是计算误差值 error[k]。
    # p是价格的数组,计算索引为 k 和索引为 k+1 之间价格的对数收益率,
    # 然后进行向量的点乘，其中 x.T 是 x 的转置，表示一个 1x2 的行向量，
    # 而 aa[:, :, k-1] 表示 aa 的第三维中索引为 k-1 的切片。
    # 这个点乘相当于将 x 与 aa[:, :, k-1] 的每一列进行对应元素的乘积，然后将结果相加
    error[k] = (np.log(p[k+1]/p[k]) - np.dot(x.T, aa[:, :, k-1])).item()  # 这一步将得到的误差值(张量格式)转换为标量值

    K = np.dot(P, x) / (np.dot(np.dot(x.T, P), x) + lmd)
    aa[:, :, k] = aa[:, :, k-1] + np.dot(K, error[k])
    P = (P - np.dot(np.dot(K, x.T), P)) / lmd

aaup = np.zeros(n)
aadn = np.zeros(n)

# az = np.zeros(n2 - n1)
# mood = np.zeros(n2 - n1)
# mdp = np.zeros(n2 - n1)
# mdn = np.zeros(n2 - n1)
# aaupp = np.zeros(n2 - n1)
# aaupn = np.zeros(n2 - n1)
# aadnp = np.zeros(n2 - n1)
# aadnn = np.zeros(n2 - n1)

az = np.zeros(n2)
mood = np.zeros(n2)
mdp = np.zeros(n2)
mdn = np.zeros(n2)
aaupp = np.zeros(n2)
aaupn = np.zeros(n2)
aadnp = np.zeros(n2)
aadnn = np.zeros(n2)

for k in range(n1, n2-1):
    aaup[k] = aa[0, 0, k]
    aadn[k] = aa[1, 0, k]

    az[k - n1 + 1] = 0
    mood[k - n1 + 1] = aadn[k - n1 + 1] - aaup[k - n1 + 1]

    if mood[k - n1 + 1] > 0:
        mdp[k - n1 + 1] = mood[k - n1 + 1]
        mdn[k - n1 + 1] = 0
    else:
        mdn[k - n1 + 1] = mood[k - n1 + 1]
        mdp[k - n1 + 1] = 0

    if aaup[k - n1 + 1] > 0:
        aaupp[k - n1 + 1] = aaup[k - n1 + 1]
        aaupn[k - n1 + 1] = 0
    else:
        aaupp[k - n1 + 1] = 0
        aaupn[k - n1 + 1] = aaup[k - n1 + 1]

    if aadn[k - n1 + 1] > 0:
        aadnp[k - n1 + 1] = aadn[k - n1 + 1]
        aadnn[k - n1 + 1] = 0
    else:
        aadnp[k - n1 + 1] = 0
        aadnn[k - n1 + 1] = aadn[k - n1 + 1]
# aaup[0] = 0
# aadn[0] = 0
# aaup = np.insert(aaup, 0, 0)
# aadn = np.insert(aadn, 0, 0)
# aaup = aaup.reshape(-1, 1)  # 转置，但这里不需要转置，因为reshape已经将其变为一列
# aadn = -aadn.reshape(-1, 1)  # 转置并取反
#
# # 对a2取反
# a2 = -a2
#
# # 创建图形窗口和子图
# plt.figure(2)
#
# # 第一个子图
# plt.subplot(2, 1, 1)
# plt.plot(aaup, 'blue')
# plt.plot(a1, 'red')
# plt.plot(aadn, 'blue')
# plt.plot(a2, 'red')
#
# # 第二个子图
# plt.subplot(2, 1, 2)
# plt.plot(p)
#
# # 显示图形
# plt.show()


# 假设 mood 已经在之前的代码中定义并赋值
# 假设 n1 和 n2 是循环的起始和结束索引

# 初始化输出数组，长度应该与循环次数匹配
# avmood = np.zeros(n2 - n1 - 5)  # 因为循环从 n1+5 开始，所以数组长度是 n2-n1-5
# avmdp = np.zeros(n2 - n1 - 5)
# avmdn = np.zeros(n2 - n1 - 5)
avmood = np.zeros(n2)  # 因为循环从 n1+5 开始，所以数组长度是 n2-n1-5
avmdp = np.zeros(n2)
avmdn = np.zeros(n2)
# 循环遍历索引范围
for k in range(n1 + 4, n2-1):  # Python中的索引从0开始，并且范围不包括结束值，所以加4来对应MATLAB的n1+5
    sum_mood = 0
    for i in range(1, 6):  # 计算前5个值的和
        sum_mood += mood[k - i]  # Python中的索引从0开始，所以不需要加1
    avmood[k - n1 - 4] = sum_mood / 5  # 将平均值存储在avmood中，并调整索引

    if avmood[k - n1 - 4] > 0:
        avmdp[k - n1 - 4] = avmood[k - n1 - 4]
        avmdn[k - n1 - 4] = 0
    else:
        avmdn[k - n1 - 4] = avmood[k - n1 - 4]
        avmdp[k - n1 - 4] = 0

    # 现在 avmood, avmdp, 和 avmdn 数组包含了转换后的结果
# 初始化交易数组
ho = 0
nb = 0  # 买入次数
ns = 0  # 卖出次数
buy = []  # 买入时间点
sel = []  # 卖出时间点

# 交易逻辑
for k in range(n1 + 5, n2-1):
    if avmood[k] > 0 and ho == 0:
        nb += 1
        buy.append(k - n1 + 2)
        ho = 1
    if avmood[k] < 0 and ho == 1:
        ns += 1
        sel.append(k - n1 + 2)
        ho = 0

    # 处理未卖出的情况
if nb > ns:
    sel.append(n2 - n1 + 1)

# 计算总收益率
p_buy = p[np.array(buy) + n1 - 1]
p_sel = p[np.array(sel) + n1 - 1]
rall = np.prod(1 + (p_sel - p_buy) / p_buy)
rall = (rall - 1) * 1000
rall = round(rall) * 0.1

# 计算买入持有收益率
rhod = 1000 * (p[n2 - 1] - p[n1 - 1]) / p[n1 - 1]
rhod = round(rhod) * 0.1

# 计算每次交易的收益率
bsp = 100 * (p_sel - p_buy) / p_buy

# 打印交易详情长度
print(len(bsp))

# 假设mm是一个索引值，从hkname中选择相应的股票名称
stock_name = 'test'

# 绘制价格和交易信号
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(range(n1, n2 + 1), p[n1 - 1:n2])
for i in range(nb):
    plt.plot([buy[i], buy[i]], [p.min(), p.max()], 'g')
for i in range(nb):
    plt.plot([sel[i], sel[i]], [p.min(), p.max()], 'r')
    plt.plot([buy[i], sel[i]], [p.min(), p.min()], 'g')
    plt.plot([buy[i], sel[i]], [p.max(), p.max()], 'g', linestyle='--')
plt.ylim([p.min() - (p.max() - p.min()) / 20, p.max() + (p.max() - p.min()) / 20])
plt.xlim([n1 - 1, n2])
#plt.text(n1 - 2, p.min() - (p.max() - p.min()) / 6, dir[n1 - 1], fontsize=12)
#plt.text(n2 - 1, p.min() - (p.max() - p.min()) / 6, dir[n2 - 1], fontsize=12)
#plt.title(
#    f'HK {stock_name} daily closing price, {dir[n1 - 1]} to {dir[n2 - 1]}, green=buy, red=sell, return= {rall:.1f}%, buy&hold= {rhod:.1f}%')

# 假设avmdp和avmdn是情绪指标的移动平均，且已经被计算
# 绘制情绪指标的移动平均
plt.subplot(2, 1, 2)
plt.plot(range(n1, n2 + 1), avmdp[n1 - 1:n2], color='g', linewidth=1.5)
plt.plot(range(n1, n2 + 1), avmdn[n1 - 1:n2], color='r', linewidth=1.5)
# 绘制情绪指标的移动平均
plt.axhline(y=0, linewidth=1.5)
#plt.text(n1 - 2, -0.125, dir[n1 - 1], fontsize=12)
#plt.text(n2 - 1, -0.125, dir[n2 - 1], fontsize=12)
plt.ylim([-0.1, 0.1])
plt.xlim([n1 - 1, n2])
plt.title('5-day moving average of mood(t) = a7(t) - a6(t); positive=buy mood, negative=sell mood')

# 显示图形
plt.show()

# 清除bsp变量（如果需要）
# del bsp

# 注意：由于Python中没有与MATLAB中的mm完全对应的命令，所以mm的使用取决于您的具体需求。
# 如果mm是一个命令或者一个需要在后面执行的特定步骤，您可能需要在Python中实现相应的功能。

# 在Python中，您可能还需要使用pandas库来处理时间序列数据，或者yfinance等库来获取股票价格数据。
# 这取决于您具体的项目需求和上下文。