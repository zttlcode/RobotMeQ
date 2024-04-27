import matplotlib.pyplot as plt
import pandas as pd
import RMQData.Indicator as RMQIndicator
from matplotlib.ticker import MultipleLocator
from mplfinance.original_flavor import candlestick2_ochl
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ohlc

from RMQTool import Tools as RMTTools

"""
matplotlib本地画图，暂时用不到了
"""


def draw_candle_orders(backtest_bar, back_test_result, isShow):
    if len(back_test_result) != 0:
        # 计算每单收益
        orders_df = pd.DataFrame(back_test_result).T  # DataFrame之后是矩阵样式，列标题是字段名，行标题是每个订单，加T是转置，列成了每单，跟excel就一样了
        print(orders_df.loc[:, 'pnl'].sum())  # 显示总收益
        # 是否图表展示
        if isShow:
            orders_df.loc[:, 'pnl'].plot.bar()  # 显示收益图

            # matplotlib 画出5分钟bar的蜡烛图
            bar_5min = pd.read_csv(backtest_bar, parse_dates=['time'], index_col=None)
            bar_5min.loc[:, 'time'] = [date2num(x) for x in bar_5min.loc[:, 'time']]  # 把日期转为float格式，才能用蜡烛图
            fig, ax = plt.subplots()  # 创建图表
            candlestick_ohlc(
                ax, bar_5min.values, width=0.2, colorup='r', colordown='green', alpha=1.0
            )

            # 把买卖点和bar图结合起来
            for index, row in orders_df.iterrows():
                ax.plot(
                    [row['openDateTime'], row['closeDateTime']],
                    [row['openPrice'], row['closePrice']],
                    color='darkblue',
                    marker='o'
                )
            plt.show()


def draw_MA_volume(DataFrame):
    """
    DataFrame展示为蜡烛图+成交量
    """
    # 设置大小，共享x坐标轴：不仅设置了绘图区域的大小，还通过sharex=True语句设置了axPrice和axVol这两个子图共享的x轴。
    figure, (axPrice, axVol) = plt.subplots(2, sharex=True, figsize=(15, 8))
    # 调用方法，绘制K线图
    candlestick2_ochl(ax=axPrice, opens=DataFrame["open"].values, closes=DataFrame["close"].values,
                      highs=DataFrame["high"].values, lows=DataFrame["low"].values,
                      width=0.75, colorup='red', colordown='green')
    axPrice.set_title("K线图和均线图")  # 设置子图标题
    # 下面6行代码，由于是在K线图和均线图的axPrice子图中操作，因此若干方法的调用主体是axPrice对象，而不是之前的pyplot.plt对象。
    DataFrame['close'].rolling(window=5).mean().plot(ax=axPrice, color="red", label='5日均线')
    DataFrame['close'].rolling(window=10).mean().plot(ax=axPrice, color="blue", label='10日均线')
    DataFrame['close'].rolling(window=20).mean().plot(ax=axPrice, color="green", label='20日均线')
    axPrice.legend(loc='best')  # 绘制图例
    axPrice.set_ylabel("价格（单位：元）")
    axPrice.grid(True)  # 带网格线
    # 如下绘制成交量子图
    # 直方图表示成交量，用for循环处理不同的颜色
    for index, row in DataFrame.iterrows():
        # 比较收盘价和开盘价，以判断当天股票是涨是跌
        if (row['close'] >= row['open']):
            # 涨了，红
            # 下面三处除以100万，结果是多少万手，对应下面成交量单位
            axVol.bar(row['time'], row['volume'] / 1000000, width=0.5, color='red')
        else:
            # 跌了，绿
            axVol.bar(row['time'], row['volume'] / 1000000, width=0.5, color='green')
    axVol.set_ylabel("成交量（单位：万手）")  # 设置y轴标题
    axVol.set_title("成交量")  # 设置子图的标题
    axVol.set_ylim(0, DataFrame['volume'].max() / 1000000 * 1.2)  # 设置y轴范围
    xmajorLocator = MultipleLocator(5)  # 将x轴主刻度设置为5的倍数
    axVol.xaxis.set_major_locator(xmajorLocator)
    axVol.grid(True)  # 带网格线
    # 旋转x轴的展示文字角度  成交量下面的日期旋转角度是15度
    for xtick in axVol.get_xticklabels():
        xtick.set_rotation(15)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def draw_MACD(DataFrame):
    """
    DataFrame展示为MACD图
    关于误差
    股票交易软件开始计算MACD指标的起始日是该股票的上市之日，而DrawMACD.py范例程序中计算的起始日是20180903，
    在这一天里，范例程序中给相关指标赋予的值仅仅是当日的指标（因为没取之前的交易数据），
    而股票交易软件计算这一天的相关指标是基于之前交易日的数据计算而来的，于是就产生了误差。
    回测数据尽量早就行了
    """
    resultDataFrame = RMQIndicator.calMACD(DataFrame, 12, 26, 9)
    # 开始绘图
    plt.figure()
    # 以折线的形式绘制出DEA和DIF两根线
    resultDataFrame['DEA'].plot(color="red", label='DEA')
    resultDataFrame['DIF'].plot(color="blue", label='DIF')
    plt.legend(loc='best')  # 绘制图例
    # 设置MACD柱状图
    for index, row in resultDataFrame.iterrows():
        if (row['MACD'] > 0):  # 大于0则用红色
            plt.bar(row['time'], row['MACD'], width=0.5, color='red')
        else:  # 小于等于0则用绿色
            plt.bar(row['time'], row['MACD'], width=0.5, color='green')
    # 设置x轴坐标的标签和旋转角度
    # 设置x轴的标签,如果显示每天的日期，那么x轴上的文字会过于密集
    # 所以只显示stockDataFrame.index%10==0（即索引值是10的倍数）的日期。
    major_index = resultDataFrame.index[resultDataFrame.index % 10 == 0]
    major_xtics = resultDataFrame['time'][resultDataFrame.index % 10 == 0]
    plt.xticks(major_index, major_xtics)
    # 设置了x轴文字的旋转角度
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # 带网格线，且设置了网格样式
    plt.grid(linestyle='-.')
    plt.title("的MACD图")
    # 显示y轴标签值里的负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def draw_MA_MACD(DataFrame):
    """
    DataFrame展示为MA+MACD图
    """
    resultDataFrame = RMQIndicator.calMACD(DataFrame, 12, 26, 9)
    # 开始绘图，设置大小，共享x坐标轴
    figure, (axPrice, axMACD) = plt.subplots(2, sharex=True, figsize=(15, 8))
    # 调用方法绘制K线图
    candlestick2_ochl(ax=axPrice, opens=resultDataFrame["open"].values, closes=resultDataFrame["close"].values,
                      highs=resultDataFrame["high"].values, lows=resultDataFrame["low"].values, width=0.75,
                      colorup='red',
                      colordown='green')
    axPrice.set_title("K线图和均线图")  # 设置子图的标题
    resultDataFrame['close'].rolling(window=3).mean().plot(ax=axPrice, color="red", label='3日均线')
    resultDataFrame['close'].rolling(window=5).mean().plot(ax=axPrice, color="blue", label='5日均线')
    resultDataFrame['close'].rolling(window=10).mean().plot(ax=axPrice, color="green", label='10日均线')
    axPrice.legend(loc='best')  # 绘制图例
    axPrice.set_ylabel("价格（单位：元）")
    axPrice.grid(linestyle='-.')  # 带网格线

    # 开始绘制第二个子图
    # 在axMACD子图内绘制了MACD线，由于是在子图内绘制，因此在绘制DEA和DIF折线的时候，需要在参数里通过“ax=axMACD”的形式指定所在的子图。
    resultDataFrame['DEA'].plot(ax=axMACD, color="red", label='DEA')
    resultDataFrame['DIF'].plot(ax=axMACD, color="blue", label='DIF')
    plt.legend(loc='best')  # 绘制图例
    # 设置第二个子图中的MACD柱状图
    for index, row in resultDataFrame.iterrows():
        if (row['MACD'] > 0):  # 大于0则用红色
            axMACD.bar(row['time'], row['MACD'], width=0.5, color='red')
        else:  # 小于等于0则用绿色
            axMACD.bar(row['time'], row['MACD'], width=0.5, color='green')
    axMACD.set_title("600895张江高科MACD")  # 设置子图的标题
    axMACD.grid(linestyle='-.')  # 带网格线
    # xmajorLocator=MultipleLocator(10)     # 将x轴的主刻度设置为10的倍数
    # axMACD.xaxis.set_major_locator(xmajorLocator)
    # 这段代码中其实给出了两种设置x轴标签的方式。如果注释掉下面两行代码，用上面两行，会发现效果是相同的。
    # 设置axMACD子图中的x轴标签，由于上面设置了axPrice和axMACD两子图共享x轴，因此K线和均线所在子图的x轴刻度会和MACD子图中的一样。
    major_xtics = resultDataFrame['time'][resultDataFrame.index % 10 == 0]
    axMACD.set_xticks(major_xtics)
    # 旋转x轴显示文字的角度
    # 因为是在子图中，所以需要通过for循环依次旋转x轴坐标的标签文字。
    for xtick in axMACD.get_xticklabels():
        xtick.set_rotation(30)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


def draw_KDJ(assetsCode, df):
    # 绘制KDJ线
    stockDataFrame = RMQIndicator.calKDJ(df)
    print(stockDataFrame)
    # 开始绘图
    plt.figure()
    stockDataFrame['K'].plot(color="blue", label='K')
    stockDataFrame['D'].plot(color="green", label='D')
    stockDataFrame['J'].plot(color="purple", label='J')
    plt.legend(loc='best')  # 绘制图例
    # 设置x轴坐标的标签和旋转角度
    major_index = stockDataFrame.index[stockDataFrame.index % 10 == 0]
    major_xtics = stockDataFrame['time'][stockDataFrame.index % 10 == 0]
    plt.xticks(major_index, major_xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # 带网格线，且设置了网格样式
    plt.grid(linestyle='-.')
    plt.title(assetsCode + "的KDJ图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def draw_KDJ_K(df):
    stockDataFrame = RMQIndicator.calKDJ(df)
    # 创建子图
    figure = plt.figure()
    (axPrice, axKDJ) = figure.subplots(2, sharex=True)
    # 调用方法，在axPrice子图中绘制K线图
    candlestick2_ochl(ax=axPrice, opens=stockDataFrame["open"].values, closes=stockDataFrame["close"].values,
                      highs=stockDataFrame["high"].values, lows=stockDataFrame["low"].values, width=0.75, colorup='red',
                      colordown='green')
    axPrice.set_title("K线图和均线图")  # 设置子图标题
    stockDataFrame['close'].rolling(window=3).mean().plot(ax=axPrice, color="red", label='3日均线')
    stockDataFrame['close'].rolling(window=5).mean().plot(ax=axPrice, color="blue", label='5日均线')
    stockDataFrame['close'].rolling(window=10).mean().plot(ax=axPrice, color="green", label='10日均线')
    axPrice.legend(loc='best')  # 绘制图例
    axPrice.set_ylabel("价格（单位：元）")
    axPrice.grid(linestyle='-.')  # 带网格线
    # 在axKDJ子图中绘制KDJ
    stockDataFrame['K'].plot(ax=axKDJ, color="blue", label='K')
    stockDataFrame['D'].plot(ax=axKDJ, color="green", label='D')
    stockDataFrame['J'].plot(ax=axKDJ, color="purple", label='J')
    plt.legend(loc='best')  # 绘制图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    axKDJ.set_title("KDJ图")  # 设置子图的标题
    axKDJ.grid(linestyle='-.')  # 带网格线
    # 设置x轴坐标的标签和旋转角度
    major_index = stockDataFrame.index[stockDataFrame.index % 5 == 0]
    major_xtics = stockDataFrame['time'][stockDataFrame.index % 5 == 0]
    plt.xticks(major_index, major_xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()


def draw_RSI(df):
    # 由于本范例程序在计算收盘价涨数和均值和收盘价跌数和均值时，用的是简单移动平均算法，因此绘制出来的图形可能和一些股票软件中的不一致，
    # 不过趋势是相同的
    # 调用方法计算RSI
    stockDataFrame = RMQIndicator.calRSI(df)
    # print(stockDataFrame)
    # 开始绘图
    plt.figure()
    stockDataFrame['RSI6'].plot(color="blue", label='RSI6')
    stockDataFrame['RSI12'].plot(color="green", label='RSI12')
    stockDataFrame['RSI24'].plot(color="purple", label='RSI24')
    plt.legend(loc='best')  # 绘制图例
    # 设置x轴坐标的标签和旋转角度
    major_index = stockDataFrame.index[stockDataFrame.index % 10 == 0]
    major_xtics = stockDataFrame['time'][stockDataFrame.index % 10 == 0]
    plt.xticks(major_index, major_xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # 带网格线，且设置了网格样式
    plt.grid(linestyle='-.')
    plt.title("RSI效果图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 调用savefig方法把图形保存到了指定目录，请注意这条程序语句需要放在show方法之前，否则保存的图片就会是空的。
    # plt.savefig('D:\\stockData\ch10\\6005842018-09-012019-05-31.png')
    plt.show()


def draw_K_MA_RSI(df):
    # 调用方法计算RSI
    stockDataFrame = RMQIndicator.calRSI(df)
    figure = plt.figure()
    # 创建子图
    (axPrice, axRSI) = figure.subplots(2, sharex=True)
    # 调用方法，在axPrice子图中绘制K线图
    candlestick2_ochl(ax=axPrice, opens=df["open"].values, closes=df["close"].values, highs=df["high"].values,
                      lows=df["low"].values, width=0.75, colorup='red', colordown='green')
    axPrice.set_title("K线图和均线图")  # 设置子图标题
    stockDataFrame['close'].rolling(window=3).mean().plot(ax=axPrice, color="red", label='3日均线')
    stockDataFrame['close'].rolling(window=5).mean().plot(ax=axPrice, color="blue", label='5日均线')
    stockDataFrame['close'].rolling(window=10).mean().plot(ax=axPrice, color="green", label='10日均线')
    axPrice.legend(loc='best')  # 绘制图例
    axPrice.set_ylabel("价格（单位：元）")
    axPrice.grid(linestyle='-.')  # 带网格线
    # 在axRSI子图中绘制RSI图形
    stockDataFrame['RSI6'].plot(ax=axRSI, color="blue", label='RSI6')
    stockDataFrame['RSI12'].plot(ax=axRSI, color="green", label='RSI12')
    stockDataFrame['RSI24'].plot(ax=axRSI, color="purple", label='RSI24')
    plt.legend(loc='best')  # 绘制图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    axRSI.set_title("RSI图")  # 设置子图的标题
    axRSI.grid(linestyle='-.')  # 带网格线
    # 设置x轴坐标的标签和旋转角度
    major_index = stockDataFrame.index[stockDataFrame.index % 7 == 0]
    major_xtics = stockDataFrame['time'][stockDataFrame.index % 7 == 0]
    plt.xticks(major_index, major_xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()
    #plt.savefig('D:\\stockData\ch10\\600584RSI.png')


def drawDonChannel(stockDf):
    """
    ·上阻力线=过去N天的最高价
    ·下支撑线=过去N天的最低价
    ·中心线=（上阻力线+下支撑线）÷2
    :param stockDf:
    :return:
    """
    # 读数据
    fig, ax = plt.subplots()
    candlestick2_ochl(ax=ax, opens=stockDf["open"].values, closes=stockDf["close"].values, highs=stockDf["high"].values,
                      lows=stockDf["low"].values, width=0.75, colorup='red', colordown='green')
    stockDf['up'] = stockDf['high'].rolling(window=20).max()
    stockDf['up'].plot(color="green", label='上阻力线')
    stockDf['down'] = stockDf['low'].rolling(window=20).min()
    stockDf['down'].plot(color="navy", label='下支撑线')
    stockDf['mid'] = (stockDf['up'] + stockDf['down']) / 2
    stockDf['mid'].plot(color="red", label='中心线')
    ax.set_ylabel("收盘价（元）")
    ax.grid()  # 带网格线
    ax.legend()  # 绘制图例
    # 设置x轴文字间隔和旋转角度
    index = stockDf.index[stockDf.index % 7 == 0]
    xtics = stockDf['time'][stockDf.index % 7 == 0]
    plt.xticks(index, xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("20天唐奇安通道效果图")
    plt.show()


def drawBollingerBands(stockDf):
    fig, ax = plt.subplots()
    candlestick2_ochl(ax=ax, opens=stockDf["open"].values, closes=stockDf["close"].values, highs=stockDf["high"].values,
                      lows=stockDf["low"].values, width=0.75, colorup='red', colordown='green')
    stockDf['mid'] = stockDf['close'].rolling(window=20).mean()
    stockDf['std'] = stockDf['close'].rolling(window=20).std()
    stockDf['up'] = stockDf['mid'] + 2 * stockDf['std']
    stockDf['down'] = stockDf['mid'] - 2 * stockDf['std']
    stockDf['up'].plot(color="green", label='上阻力线')
    stockDf['down'].plot(color="navy", label='下支撑线')
    stockDf['mid'].plot(color="red", label='中心线')
    ax.set_ylabel("收盘价（元）")
    ax.grid()  # 带网格线
    ax.legend()  # 绘制图例
    # 设置x轴文字间隔和旋转角度
    index = stockDf.index[stockDf.index % 7 == 0]
    xtics = stockDf['time'][stockDf.index % 7 == 0]
    plt.xticks(index, xtics)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("20天布林带通道效果图")
    plt.show()


if __name__ == '__main__':
    filePath = RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_601012_d.csv'
    DataFrame = pd.read_csv(filePath, encoding='gbk')
    drawBollingerBands(DataFrame)