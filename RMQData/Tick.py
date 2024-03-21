from datetime import timedelta
import pandas as pd
import numpy as np
import os
from dateutil import parser


def assertCode(assetsCode, assetsType):
    code = None
    # 只有一个特殊：0开头的指数，是sh，代码没加这种情况
    # 因为我目前不做指数，etf是5、1开头
    if assetsCode.startswith('5') or assetsCode.startswith('6'):
        code = 'sh' + assetsCode
    elif assetsCode.startswith('0') or assetsCode.startswith('1') or assetsCode.startswith('3'):
        code = 'sz' + assetsCode
        if assetsType == 'index' and assetsCode.startswith('0'):
            code = 'sh' + assetsCode
    return code


def getTick(req, assetsCode, assetsType):
    code = assertCode(assetsCode, assetsType)
    page = req.get("https://web.sqt.gtimg.cn/q="+code, timeout=3.05)
    # 原来这个数据是f12 在network里找到的
    # https://web.sqt.gtimg.cn/q=sz002714
    # http://qt.gtimg.cn/q=sz002714
    # 获取返回报文
    stock_info_response = page.text
    # 分割报文内容
    stock_info = stock_info_response.split("~")
    # 提取股票收盘价
    current_price = float(stock_info[3])
    # 提取交易日期
    trade_dateTime = stock_info[30]
    # 提取成交量 手
    trade_volume = float(stock_info[6])
    # 收盘价、交易时间  放入元组 , 把时间从字符串转为date格式
    tick = (parser.parse(trade_dateTime), current_price, trade_volume)  # parser.parse(trade_dateTime).time()
    return tick


def getTickForDay(req, assetsCode, assetsType):
    code = assertCode(assetsCode, assetsType)
    page = req.get("https://web.sqt.gtimg.cn/q="+code, timeout=3.05)
    # 获取返回报文
    stock_info_response = page.text
    # 分割报文内容
    stock_info = stock_info_response.split("~")
    # 提取股票开盘价
    open_price = float(stock_info[5])
    # 提取股票最高价
    high_price = float(stock_info[33])
    # 提取股票最低价
    low_price = float(stock_info[34])
    # 提取股票收盘价
    close_price = float(stock_info[3])
    # 提取交易日期
    trade_dateTime = stock_info[30]
    # 提取成交量 手
    trade_volume = float(stock_info[6])
    # 收盘价、交易时间  放入元组 , 把时间从字符串转为date格式
    tick = (parser.parse(trade_dateTime), open_price, high_price, low_price, close_price, trade_volume)
    return tick


def trans_bar_to_ticks(code, timeLevel, bar_path, ticks):
    # 此函数实盘不用，回测或实盘前加载历史数据用，因此成交量只放最后一个tick里，最终在bar_generator更新volume
    # 读取历史bar数据，
    bar_data = pd.read_csv(bar_path, parse_dates=['time'])
    '''  为什么bar要转tick
    假设我的策略是以5分钟bar为基础运行的，在终端里，只有这5分钟结束后对应
    的bar数据才会返回。但我觉得等5分钟bar返回时再进入买卖逻辑时黄花菜都凉了
    ，不如每返回一个tick就更新一次5分钟bar，这样我得到的5分钟bar虽然
    是个“残缺的”bar，但它涵盖了标的最新信息。
    
     数据处理思路：
         每一条历史数据里有交易时间、成交量、四个价格。
         我需要的是像实盘一样，价格一点点变化，就像一个数组。这个数组里是价格变化的过程
         价格变化的过程，不论如何反复，总能抽象成这四个价格。
         所以，我从开盘价，以某个步长，加到最高价，再从开盘价以某个步长，减到最低价，再补上收盘价，构成价格变化数组
     '''
    for index, row in bar_data.iterrows():  # 日线数据用pandas封装成DataFrame对象，逐行遍历。index是每行数据行号，row是每行内容
        # 这个循环里的操作目的，是把每行的4个价格，扩展成一个模拟价格变化的数组
        if code.startswith('5') or code.startswith('1'):
            # 5,1开头，说明是etf
            step = 0.001
        else:
            # 其他就是股票或指数，都是两位小数
            if row['open'] < 30:  # 控制数组的价格步长，如果价格太高，步长太小，会导致数组过大，浪费空间
                step = 0.01
            elif row['open'] < 100:
                step = 0.05
            elif 100 < row['open'] < 1000:
                step = 0.1
            elif 1000 < row['open'] < 10000:
                step = 1
            else:
                step = 2
        # 确定好步长后，组装价格变化数组。numpy包的一大功能就是给个开头结尾，人家给你补上中间的数组
        arr = np.arange(row['open'], row['high'], step)
        arr = np.append(arr, row['high'])  # 步长的加法有时无法显示结尾数字，这里手动加上，以免它自己漏了，如果没漏，重复一下也无所谓
        arr = np.append(arr, np.arange(row['open'] - step, row['low'], -step))  # 再补从开盘到最低价的数据
        arr = np.append(arr, row['low'])  # 同high
        arr = np.append(arr, row['close'])  # 最后补个收盘价  这个五分钟的模拟数组就做好了

        # 拼接好数组之后，给每个元素配个交易时间，使价格、交易时间，组成一个元组，再把元组放到一个列表里
        i = 0  # 用于把叠加时间
        for item in arr:  # 给数组里的每个元素配个交易时间
            if i == len(arr) - 1:
                # 当前是最后一个tick，要带bar的成交量，代表这个bar一共多少成交量
                # 日线数据
                if row['time'].hour == 0:
                    ticks.append((row['time'], item, row['volume']))  # 日线数据 2022-01-05
                else:
                    # 分钟数据
                    if timeLevel == "5":
                        ticks.append((row['time'] - timedelta(minutes=5) + timedelta(seconds=0.1 * i), item,
                                      row['volume']))  # 分钟线数据 2022-01-10 09:40:00
                    elif timeLevel == "15":
                        ticks.append((row['time'] - timedelta(minutes=15) + timedelta(seconds=0.1 * i), item,
                                      row['volume']))  # 分钟线数据 2022-01-10 09:40:00
                    elif timeLevel == "30":
                        ticks.append((row['time'] - timedelta(minutes=30) + timedelta(seconds=0.1 * i), item,
                                      row['volume']))  # 分钟线数据 2022-01-10 09:40:00
                    elif timeLevel == "60":
                        ticks.append((row['time'] - timedelta(hours=1) + timedelta(seconds=0.1 * i), item,
                                      row['volume']))  # 分钟线数据 2022-01-10 09:40:00
            else:
                # 当前tick不是最后一个，不需要带成交量，置为0
                # 日线数据
                if row['time'].hour == 0:
                    ticks.append((row['time'], item, 0))  # 列表里存放的是一个个元组
                else:
                    # 分钟数据
                    if timeLevel == "5":
                        ticks.append((row['time'] - timedelta(minutes=5) + timedelta(seconds=0.1 * i), item, 0))
                    elif timeLevel == "15":
                        ticks.append((row['time'] - timedelta(minutes=15) + timedelta(seconds=0.1 * i), item, 0))
                    elif timeLevel == "30":
                        ticks.append((row['time'] - timedelta(minutes=30) + timedelta(seconds=0.1 * i), item, 0))
                    elif timeLevel == "60":
                        ticks.append((row['time'] - timedelta(hours=1) + timedelta(seconds=0.1 * i), item, 0))
            i += 1
    # 至此，tick列表里，放了一堆元组，每个元组里都是价格和时间，模拟了一段日子里，某个指数所有每天的价格变化
    return ticks


def get_ticks_for_backTesting(assetsCode, backtest_tick, backtest_bar, timeLevel):
    if os.path.exists(backtest_tick):
        # 1、读取数据：回测数据已经生成过了
        ticks = pd.read_csv(backtest_tick,
                            parse_dates=['datetime'],  # csv中此列是字符串格式，这里转为时间格式
                            index_col='datetime')  # 把此列当作第一列
        # 2、转换格式：ticks数据是pandas,要转a为numpy
        tick_list = []
        for index, row in ticks.iterrows():  # 把读取的list数据遍历出来
            tick_list.append((index, row[0], row[1]))  # 把每一行元素的索引（这里是datetime)和行里第1、2个元素放入list列表
        ticks = np.array(tick_list)  # 把list转成np类型
    else:
        # 1、生成数据：回测数据还没有
        ticks = []
        ticks = trans_bar_to_ticks(assetsCode, timeLevel, backtest_bar, ticks)
        # 2、数据保存本地
        tick_DataFrame = pd.DataFrame(ticks, columns=['datetime', 'price', 'volume'])
        tick_DataFrame.to_csv(backtest_tick, index=0)
    return ticks

