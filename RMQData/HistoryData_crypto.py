import pandas as pd
import math
import datetime
from binance.spot import Spot
from RMQTool import Tools as RMTTools


def binance_to_his_csv(client, quot, remainder, k_interval):
    # 用来拼接df
    result_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'time'])

    k_tmp_start_date = start_date

    if quot > 0:  # 说明得调多次
        if k_interval == '1d':
            k_tmp_end_date = k_tmp_start_date + datetime.timedelta(days=1000)  # 加1000天
        elif k_interval == '4h':
            k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=4000)
        elif k_interval == '1h':
            k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=1000)
        elif k_interval == '15m':
            k_tmp_end_date = k_tmp_start_date + datetime.timedelta(minutes=15 * 1000)

        i = 1
        while i <= quot:
            df = pd.DataFrame(
                client.klines(symbol="BTCUSDT", interval=k_interval, limit=1000,
                              startTime=int(k_tmp_start_date.timestamp() * 1000),
                              endTime=int(k_tmp_end_date.timestamp() * 1000)))
            data = df.iloc[:, 1:7]  # 含头不含尾，第1列到第6列
            data.columns = ['open', 'high', 'low', 'close', 'volume', 'time']
            data.drop_duplicates(subset=['time'], keep="first", inplace=True)  # 按time列删除重复，只保留重复的第一个
            data['time'] = data['time'] + 1  # 以结束时间做时间，加了1毫秒凑个整时间

            # 拼接df
            result_df = pd.concat([result_df, data])
            i += 1
            k_tmp_start_date = k_tmp_end_date
            if i > quot:
                break
            else:
                if k_interval == '1d':
                    k_tmp_end_date = k_tmp_start_date + datetime.timedelta(days=1000)  # 加1000天
                elif k_interval == '4h':
                    k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=4 * 1000)
                elif k_interval == '1h':
                    k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=1000)
                elif k_interval == '15m':
                    k_tmp_end_date = k_tmp_start_date + datetime.timedelta(minutes=15 * 1000)

    if k_interval == '1d':
        k_tmp_end_date = k_tmp_start_date + datetime.timedelta(days=remainder)  # 加余数的天
    elif k_interval == '4h':
        k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=4 * remainder)
    elif k_interval == '1h':
        k_tmp_end_date = k_tmp_start_date + datetime.timedelta(hours=remainder)
    elif k_interval == '15m':
        k_tmp_end_date = k_tmp_start_date + datetime.timedelta(minutes=15 * remainder)
    try:
        df = pd.DataFrame(
            client.klines(symbol="BTCUSDT", interval=k_interval, limit=remainder,
                          startTime=int(k_tmp_start_date.timestamp() * 1000),
                          endTime=int(k_tmp_end_date.timestamp() * 1000)))
    except Exception as e:
        print("Error happens", e, k_interval, remainder, k_tmp_start_date, k_tmp_end_date)

    data = df.iloc[:, 1:7]  # 含头不含尾，截取第3行到倒数第二行，第0列到第5列
    data.columns = ['open', 'high', 'low', 'close', 'volume', 'time']
    data.drop_duplicates(subset=['time'], keep="first", inplace=True)  # 按time列删除重复，只保留重复的第一个
    data['time'] = data['time'] + 1  # 以结束时间做时间，加了1毫秒凑个整时间

    # 拼接df
    result_df = pd.concat([result_df, data], ignore_index=True)
    result_df['time'] = pd.to_datetime(result_df['time'], unit='ms')
    # 把时间列调整到第一列
    result_df_tmp = result_df.pop('time')
    result_df.insert(0, 'time', result_df_tmp)
    # 把时间设为index
    result_df.set_index('time', inplace=True)

    path = RMTTools.read_config("RMQData", "backtest_bar")
    # 把结果df写入csv
    if k_interval == '1d':
        result_df.to_csv(path + "backtest_bar_crypto_BTCUSDT_d.csv")
        print("日线数据写入完成")
    elif k_interval == '4h':
        result_df.to_csv(path + "backtest_bar_crypto_BTCUSDT_240.csv")
        print("4h数据写入完成")
    elif k_interval == '1h':
        result_df.to_csv(path + "backtest_bar_crypto_BTCUSDT_60.csv")
        print("1h数据写入完成")
    elif k_interval == '15m':
        result_df.to_csv(path + "backtest_bar_crypto_BTCUSDT_15.csv")
        print("15m数据写入完成")


def get_history_crypto(start_date, end_date):
    proxies = {
        'http': 'http://127.0.0.1:33210',
        'https': 'http://127.0.0.1:33210',
    }
    client = Spot(proxies=proxies)
    # client = Spot()

    # 计算时间段差了多少天
    diff = (end_date - start_date).days

    # 计算各个级别，差了多少
    """
    :param diff: 时间段差了几天
    :return: 返回字典，其中 quot个1000bar，和 remainder个bar

    {'day':{'quot':quot,'remainder':remainder},
    'hour':{'quot':quot,'remainder':remainder},
    'minute':{'quot':quot,'remainder':remainder}}
    """
    dic = {}
    dic['1d'] = {'quot': math.floor(diff / 1000), 'remainder': diff % 1000}
    dic['4h'] = {'quot': math.floor(diff * 6 / 1000), 'remainder': diff * 6 % 1000}
    dic['1h'] = {'quot': math.floor(diff * 24 / 1000), 'remainder': diff * 24 % 1000}
    dic['15m'] = {'quot': math.floor(diff * 24 * 60 / 15 / 1000), 'remainder': diff * 24 * 60 / 15 % 1000}
    # 需要其他级别，在这里直接加

    # 给每个级别，取数并拼接
    binance_to_his_csv(client, dic['1d']['quot'], dic['1d']['remainder'], '1d')
    binance_to_his_csv(client, dic['4h']['quot'], dic['4h']['remainder'], '4h')
    binance_to_his_csv(client, dic['1h']['quot'], dic['1h']['remainder'], '1h')
    binance_to_his_csv(client, dic['15m']['quot'], dic['15m']['remainder'], '15m')


if __name__ == '__main__':
    """
    获取历史数据
        历史数据官网是一条一条存储的，需要下载，再拼接
        我想那就不直接用接口调，1000条一拼接，从2020年4.21到2023年4.21
        每个级别都试，15m、1h、4h、day 再30m 2h、8h、  再5m、6h
        输入时间范围，级别；算出总共多少秒，算出有多少个对应的级别，换算成时间段，最后能自动拼接好数据
    """
    start_date = datetime.datetime(2012, 4, 16)
    end_date = datetime.datetime(2024, 4, 16)
    get_history_crypto(start_date, end_date)
