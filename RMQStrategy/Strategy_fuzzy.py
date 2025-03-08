import time
import os
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
import pandas as pd

from RMQTool import Tools as RMTTools
import RMQData.Position as RMQPosition
import RMQTool.Run_live_model as RMQRun_live_model


def meb(x, w1, w2, w3):
    # 定义fuzzy set 7个μ的分段函数，叫做隶属函数
    # x是x*t，w是幅度也是x轴刻度
    y = 0
    if x <= w1:
        y = 0
    if w1 < x <= w2:
        y = (x - w1) / (w2 - w1)
    if w2 < x <= w3:
        y = (w3 - x) / (w3 - w2)
    if x > w3:
        y = 0

    if w1 == w2:
        if x <= w2:
            y = 1
        if w2 < x <= w3:
            y = (w3 - x) / (w3 - w2)
        if x > w3:
            y = 0

    if w2 == w3:
        if x <= w1:
            y = 0
        if w1 < x <= w2:
            y = (x - w1) / (w2 - w1)
        if x > w2:
            y = 1

    return y  # 返回μ值


def validMeb():
    # 以论文part2的函数代入为例，试验隶属函数，发现函数代入与我直接用，结果是一样的
    x = Decimal('-0.04')
    w = Decimal('0.01')
    x_list = []
    y_list = []
    for i in range(80):
        y1 = meb(x, 0, w, 2 * w)  # PS
        y2 = meb(x, w, 2 * w, 3 * w)  # PM
        y3 = meb(x, 2 * w, 3 * w, 3 * w)  # PL
        y4 = meb(x, -2 * w, -w, 0)  # NS
        y5 = meb(x, -3 * w, -2 * w, -w)  # NM
        y6 = meb(x, -3 * w, -3 * w, -2 * w)  # NL
        y7 = meb(x, -w, 0, w)  # AZ

        ed_0 = 0
        y_div_0 = y1 + y2 + y3 + y4 + y5 + y6 + y7
        if y_div_0 != 0:
            ed_0 = (Decimal('0.1') * y1 + Decimal('0.4') * y2 - Decimal('0.2') * y3 - Decimal('0.1') * y4 - Decimal(
                '0.4') * y5 + Decimal('0.2') * y6 + Decimal('0') * y7) / y_div_0
        x += Decimal('0.001')
        if x == Decimal('0.02'):
            print(x)
        x_list.append(round(x, 3))
        y_list.append(round(ed_0, 3))

    plt.plot(np.array(x_list), np.array(y_list))
    plt.show()


def fuzzy(windowDF, bar_num):
    # 1、计算自己需要的指标
    # windowDF = RMQIndicator.calMA(windowDF)
    p = windowDF['close'].tolist()
    n = bar_num  # 250不一定够
    c = 0.01
    n1 = 1
    n2 = n
    ma1 = 5
    lmd = 0.95
    P = np.eye(2) * windowDF.iloc[0]['close']  # 声明单位矩阵
    aa = np.zeros((2, 1, n))  # 2个 1*n的向量，用来记录两个参数在n个时刻的值
    error = np.zeros(n)

    for k in range(n1 + ma1):
        aa[:, :, k] = np.array([[0], [0]])
        # 0~3初始化为0

    for k in range(n1 + ma1, n2 - 1):
        try:
            # 循环到n-1,但下面的p会取到最新价格k+1,也就是n2
            pa = np.sum(p[k - ma1:k]) / ma1
            x3 = np.log(p[k] / pa)
            y1 = meb(x3, 0, c, 2 * c)
            y2 = meb(x3, c, 2 * c, 3 * c)
            y3 = meb(x3, 2 * c, 3 * c, 3 * c)
            y4 = meb(x3, -2 * c, -c, 0)
            y5 = meb(x3, -3 * c, -2 * c, -c)
            y6 = meb(x3, -3 * c, -3 * c, -2 * c)
            y7 = meb(x3, -c, 0, c)
            y = y1 + y2 + y3 + y7
            ed1 = 0
            if y != 0:
                ed1 = (-0.1 * y1 - 0.2 * y2 - 0.4 * y3) / y
            y = y4 + y5 + y6 + y7
            ed2 = 0
            if y != 0:
                ed2 = (0.1 * y4 + 0.2 * y5 + 0.4 * y6) / y

            x = np.array([[ed1], [ed2]])

            # 这行代码的目的是计算误差值 error[k]。
            # p是价格的数组,计算索引为 k 和索引为 k+1 之间价格的对数收益率,
            # 然后进行向量的点乘，其中 x.T 是 x 的转置，表示一个 1x2 的行向量，
            # 而 aa[:, :, k-1] 表示 aa 的第三维中索引为 k-1 的切片。
            # 这个点乘相当于将 x 与 aa[:, :, k-1] 的每一列进行对应元素的乘积，然后将结果相加
            error[k] = (np.log(p[k + 1] / p[k]) - np.dot(x.T, aa[:, :, k - 1])).item()  # 这一步将得到的误差值(张量格式)转换为标量值

            K = np.dot(P, x) / (np.dot(np.dot(x.T, P), x) + lmd)
            # 最后一对系数比最新价格早一位,k截至到n2-1,最新价格是n2
            aa[:, :, k] = aa[:, :, k - 1] + np.dot(K, error[k])
            P = (P - np.dot(np.dot(K, x.T), P)) / lmd
        except Exception as e:
            print(e)
    return n1, n2, aa


def strategy_fuzzy(positionEntity,
                   indicatorEntity,
                   windowDF_calIndic,
                   bar_num,
                   strategy_result):
    # 2025 03 06 A股就不用移动止损了
    # if 0 != len(positionEntity.currentOrders):  # 满仓，判断止损
    #     RMQPosition.stopLoss(positionEntity, indicatorEntity, strategy_result)

    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            n1, n2, aa = fuzzy(windowDF_calIndic, bar_num)
            # bar_num为了算过去的指标，250-1，所以n2是249,最后一位下标248，没值，243~247有值，
            mood = aa[1, 0, n2 - 6:n2 - 1] - aa[0, 0, n2 - 6:n2 - 1]  # a7-a6的值，正：可以买
            avmood = np.mean(mood)

            # 空仓，且大买家占优则买
            if 0 == len(positionEntity.currentOrders) and avmood > 0:  # 空仓时买
                # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                               round(indicatorEntity.tick_close, 3),
                               "buy"]
                positionEntity.trade_point_list.append(trade_point)
                # # 推送消息
                # strategy_result.send_msg(indicatorEntity.IE_assetsName
                #                          + "-"
                #                          + indicatorEntity.IE_assetsCode,
                #                          indicatorEntity,
                #                          None,
                #                          "buy" + str(round(avmood, 3)))
                RMQRun_live_model.run_live_call_model(indicatorEntity, "buy")  # 2025 03 06 实盘调模型，不发消息

                volume = int(positionEntity.money / indicatorEntity.tick_close / 100) * 100
                # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
                RMQPosition.buy(positionEntity, indicatorEntity, indicatorEntity.tick_close, volume)
            # 满仓
            if 0 != len(positionEntity.currentOrders) and avmood < 0:
                # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                               round(indicatorEntity.tick_close, 3),
                               "sell"]
                positionEntity.trade_point_list.append(trade_point)
                # # 设置推送消息
                # strategy_result.send_msg(indicatorEntity.IE_assetsName
                #                          + "-"
                #                          + indicatorEntity.IE_assetsCode,
                #                          indicatorEntity,
                #                          None,
                #                          "sell" + str(round(avmood, 3)))
                RMQRun_live_model.run_live_call_model(indicatorEntity, "sell")  # 2025 03 06 实盘调模型，不发消息
                # 卖
                RMQPosition.sell(positionEntity, indicatorEntity)


def detach_coefficient_figure(p, n, n1, n2, aa):
    aaup = np.zeros(n)
    aadn = np.zeros(n)
    mood = np.zeros(n2)
    for k in range(n1, n2 - 1):
        # 注意在策略里系数是早一位的,所以截至到n2-1
        aaup[k] = aa[0, 0, k]  # 2*1矩阵，上面那行是a6，下面是a7，策略是看a7-a6的值，正：可以买，变负，就卖
        aadn[k] = aa[1, 0, k]
        mood[k - n1 + 1] = aadn[k - n1 + 1] - aaup[k - n1 + 1]

    # 初始化输出数组
    avmood = np.zeros(n2)
    avmdp = np.zeros(n2)
    avmdn = np.zeros(n2)
    # 循环遍历索引范围,操作的是系数,所以还是截至到n2-1
    for k in range(n1 + 4, n2 - 1):  # Python中的索引从0开始，并且范围不包括结束值，所以加4来对应MATLAB的n1+5
        sum_mood = 0
        for i in range(1, 6):  # 计算前5个值的和
            sum_mood += mood[k - i]  # Python中的索引从0开始，所以不需要加1
        avmood[k] = sum_mood / 5  # 第k天之前5天的系数差均值，放在第k天

        if avmood[k] > 0:
            avmdp[k] = avmood[k]
            avmdn[k] = 0
        else:
            avmdn[k] = avmood[k]
            avmdp[k] = 0
        # 现在 avmood, avmdp, 和 avmdn 数组包含了转换后的结果
    # 初始化交易数组
    ho = 0
    nb = 0  # 买入次数
    ns = 0  # 卖出次数
    buy = []  # 买入时间点
    sel = []  # 卖出时间点

    # 交易逻辑
    for k in range(n1 + 4, n2 - 1):
        # 同上, 循环遍历索引范围,操作的是系数,所以还是截至到n2-1,但买卖点是定位在最新价格,是n,所以加2
        if avmood[k] > 0 and ho == 0:
            nb += 1
            buy.append(k - n1 + 2)  # 第k天看过去5天的系数差均值，大于0，则k+1天买入  20240708删了-5，-5相当于知道涨跌，提前5天交易，不合理
            ho = 1
        if avmood[k] < 0 and ho == 1:
            ns += 1
            sel.append(k - n1 + 2)  # k - n1 + 2   20240708删了-5，
            ho = 0

    # 处理未卖出的情况
    if nb > ns:
        sel.append(n2 - n1)  # n2-n1就是list的最后一个下标,最后一刻卖出

    # 计算此策略的总收益率
    p_buy = p[np.array(buy) + n1 - 1]
    p_sel = p[np.array(sel) + n1 - 1]
    rall = np.prod(1 + (p_sel - p_buy) / p_buy)  # 计算累积收益率,+1是加上本金
    rall = (rall - 1) * 1000  # 减去本金,下面是换算成百分比
    rall = round(rall) * 0.1

    # 计算买入持有收益率
    rhod = 1000 * (p[n2 - 1] - p[n1 - 1]) / p[n1 - 1]  # 计算买入并持有策略的收益率
    rhod = round(rhod) * 0.1

    # 计算每次交易的收益率
    bsp = 100 * (p_sel - p_buy) / p_buy

    # 打印交易详情长度  每次交易收益率
    print(len(bsp), bsp)

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

    print(f'green=buy, red=sell, return= {rall:.1f}%, buy&hold= {rhod:.1f}%')

    # 假设avmdp和avmdn是情绪指标的移动平均，且已经被计算
    # 绘制情绪指标的移动平均
    plt.subplot(2, 1, 2)
    plt.plot(range(n1, n2 + 1), avmdp[n1 - 1:n2], color='g', linewidth=1.5)
    plt.plot(range(n1, n2 + 1), avmdn[n1 - 1:n2], color='r', linewidth=1.5)
    # 绘制情绪指标的移动平均
    plt.axhline(y=0, linewidth=1.5)
    plt.ylim([-0.1, 0.1])
    plt.xlim([n1 - 1, n2])
    plt.title('5-day moving average of mood(t) = a7(t) - a6(t); positive=buy mood, negative=sell mood')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    def find_files_with_char(directory, char):
        files_with_char = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if char in file:
                    files_with_char.append(os.path.join(root, file))
        return files_with_char
    """
    1、模糊之前强是因为时间错位了，知道5天后涨，所以今天买，因此利润高
    2、bug修改后指标就显得后知后觉，提示买时已经涨好几天了，经常买在最高位，跌了好几天才有卖出信号
    3、我加入了移动止损，理论上能大幅改善这个策略 
    """
    # 使用函数查找含有字符'd'的文件，并将文件名列表打印出来
    directory_path = RMTTools.read_config("RMQData", "live_bar")  # 替换为你的目录路径
    char_to_find = '_d'
    filtered_files = find_files_with_char(directory_path, char_to_find)
    for filePath in filtered_files:
        windowDF = pd.read_csv(filePath, encoding='gbk')
        bar_num = len(windowDF)
        # 回测
        n1, n2, aa = fuzzy(windowDF, bar_num)
        # n1 和 n2 是循环的起始和结束索引
        p = windowDF['close'].values
        print(filePath)
        detach_coefficient_figure(p, bar_num, n1, n2, aa)  # 图表展示回测结果
        time.sleep(1)

