import numpy as np
import pandas as pd
from math import exp
'''
最优控制：不论初始状态和初始决策如何，对于前面决策造成的某一状态而言，其后各阶段的决策序列必须构成最优策略。
具体做法是通过动态规划递推。
Hamilton-Jacobi-Bellman equation 哈密顿-雅可比-贝尔曼方程，简称HJB方程，是一个偏微分方程，他是最优控制的核心。
HJB方程的解是针对特定动态系统及相关代价函数下有最小代价的实值函数。

论文思路如下：
1、先把配对交易问题转化为随机最有控制问题
2、算出HJB方程的解
3、用模拟数据试验
4、出结论
5、给出参数估计的答案
'''
# -------------------------------------接下来是策略代码-----------------------------------------------------------------

# A_price = 资产A每个bar的收盘价组成的价格列表，是个pandas对象
# B_price = 资产B每个bar的收盘价组成的价格列表，是个pandas对象
dataFrame = pd.read_csv()  # 把A_price，B_price放在一起
data = None  # 防报错临时加的
V= [1] # V(t)表示套利资产A、资产B这个策略所代表的净值，可以看作随时间t变化的策略收益。
# 利润用V保存，初值为1，要回测，就把所有历史数据传进来;
# 要实盘，不需要V，只传最近的数据，比如传260条，窗口大小250，那循环只进行10次，最后一次的结果就是当前时刻要进行的操作

def stochastic_optimal_control_strategy(dataFrame):
    A_price = dataFrame.iloc[:, 0].values
    B_price = dataFrame.iloc[:, 1].values
    S = np.log(B_price)  #  将资产B的价格对数化
    X = np.log(A_price) - np.log(B_price)  #  表示t时刻两个资产的价差
    '''
    假设价差走势符合Ornstein-Uhlenbeck process 奥恩斯坦-乌伦贝克过程，是指一个平稳的高斯-马尔科夫过程，其数学期望为0且以指数函数为核函数。
    k*(theta-X(t))表示在t时刻预期的瞬时价差变化的漂移项，theta是价差的长期均衡水平，均值回复速度由 k 决定 (k在参数估计中由p决定)，
    eta决定价差的波动性，rho表示 B(t)和OU两个式子里的布朗运动的瞬时相关系数。
    '''
    # 设置参数
    delta_t = 1  # 观察的时间周期，比如3个bar一观察，1个bar一观察，一般都是一个bar，我不知道为啥用1，论文用 1/251；delta_t = T-t 最终时刻减去当前时刻，1-0=1
    T = 1  # 终期时间，实盘时数据没有终点
    N = 60  # 窗口大小，我觉得250比较好
    gamma = -100  #  跟效用函数有关，论文这么设置的，不知道为啥，就这么用着吧

    for t in range(N, len(data) -1):  # 从第N条数据开始，算到最后一条，每次算窗口大小为N的数据量
        # 估计参数和实际参数是一个东西，在代码里为了简洁就用实际参数的变量名
        # 整个算法设计的参数如下：
        m = ( S[t] - S[t-N] )/ N
        SS = ( np.power((pd.Series(S[t- N:t + 1]) - pd.Series(S[t - N:t + 1]).shift(1)), 2)[1:].sum() - 2* m * (S[t] - S[t-N]) + N * m ** 2 ) / N
        p = 1/ (N * np.power(X[t - N:t], 2).sum() - X[t - N:t].sum() ** 2) * (N * (pd.Series(X[t - N:t + 1]) * pd.Series(X[t - N:t + 1]).shift(1) )[1:].sum() - (X[t] - X[t-N]) *X[t- N:t].sum() -np.power(X[t- N:t].sum(), 2))
        if p<0:
            p = -p
            q = (X[t] - X[t - N] +X[t - N:t].sum() * (1 - p)) / N
            # 防报错临时加的VV = 1 / N * (X[t] **2-X[t - N] ** 2 + (1 + p ** 2) * np.power(X[t - N:t], 2).sum() – 2 * p * (pd.Series(X[t - N:t + 1]) * pd.Series(X[t - N:t +1]).shift(1))[1:].sum() - N * q)
        if VV<0:  # 二阶导<0存在极大值
            VV = -VV
            C = 1 / (N * VV ** 0.5 * SS ** 0.5) * ((pd.Series(X[t - N:t + 1]) * (pd.Series(S[t - N:t + 1]) - pd.Series(S[t - N:t + 1]).shift(1)))[1:].sum() - p * (pd.Series(X[t - N:t +1]) * (pd.Series(S[t - N:t + 1]).shift(-1) -pd.Series(S[t - N:t + 1])))[:-1].sum() - m * (X[t] -X[t - N]) - m * (1- p) * pd.Series(X[t - N:t]).sum())
            sigma = (SS/delta_t)**0.5  # sigma是资产B假定为布朗运动时的波动率
            mu = m/delta_t + 0.5 * sigma**2  # 这个是模拟价格时用的，实盘没有用到 mu是资产B假定为布朗运动时的漂移项
            k = -(np.log(p)/delta_t)
            theta = q/(1-p)
            eta  = (2*k*VV/(1-p**2))**0.5
            rho = k * C * (VV ** 0.5) * (SS ** 0.5) / (eta * sigma * (1-p))
            beta=1/ (2 * eta ** 2 * (1 - (1-gamma) ** 0.5 - (1 + (1-gamma) ** 0.5) * exp(2 * k * (T -0 ) / ((1-gamma) ** 0.5)))) * (gamma * (1-gamma) ** 0.5 * (eta ** 2 + 2 * rho * sigma * eta) * ((1- exp(2 * k * (T - 0) / ((1-gamma) ** 0.5))) ** 2) - gamma * (eta ** 2 + 2 * rho * sigma * eta +2 * k * theta) * (1 - exp(2 * k * (T - 0) / ((1-gamma) ** 0.5))))
            # 防报错临时加的alpha = k * (1 – (1-gamma) ** 0.5) / (2 * eta ** 2) * (1 + 2 * (1-gamma) ** 0.5/ (1 – (1-gamma) ** 0.5 - (1 + (1-gamma) ** 0.5) * exp(2 * k * (T - 0 ) / ((1-gamma) ** 0.5))))

            '''
            论文的目标是最大化预期效用（效用函数，经济学中的概念，引入参数gamma）。于是转换为最优控制问题，有边界条件、系统状态方程，即初始净值、初始价差、净值变化、价差变化
            最优控制问题的值函数是 G(t,v,x) t时间，v净值，x价差，最优控制问题的HJB方程也就有了。求函数极大值也就是一阶导=0，解微分方程得当前时刻最优资金权重 h(t)，
            论文用 h*(t，x)表示。
            h*(t,x)是当前时刻的最优资金权重，指导实盘仓位变动，比如h*(t,x)降低了，就是卖空A，同时做多B，下次h*(t,x)再变动就继续操作，两次之间的收益记录在V里。
            '''

            # 防报错临时加的h = 1 / (1-gamma) * (beta + 2 * X[t] * alpha- k * (X[t] - theta) / (eta ** 2) + rho * sigma / eta + 0.5)

            '''
            h(t)，h~(t)表示资产A、资产B在t时刻的组合权重
            那t时刻的净值变化就是 dV(t)=V(t)*( h(t)*dA(t)/A(t) + h~(t)*dB(t)/B(t) +dM(t)/M(t) ) 这里 M(t)是无风险资产，无风险利率是r，则有 dM(t) = r*M(t)dt，但我们实盘不讲究这个，忽略无风险利率
            因为成对交易，所以卖空一个时以相同资金做多另一个。所以 h(t) = -h~(t)，所以 dV(t)也可以写成'''

            # 防报错临时加的dV = V[-1] * (h * (A_price[t + 1] - A_price[t]) / A_price[t] - h * (B_price[t+1] - B_price[t]) / B_price[t])
            # 防报错临时加的V.append(V[-1] + dV)
            # 实盘时，注释最后两行，放开最后一行，根据h的变动，决定
            # return h

import matplotlib.pyplot as plt

'''第一幅图为价差曲线'''
# 防报错临时加的pd.Series(X).plot()
plt.show()
'''第二幅图为收益曲线'''
pd.Series(V).plot()
plt.show()