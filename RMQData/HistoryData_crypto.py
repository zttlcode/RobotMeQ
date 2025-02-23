import pandas as pd
import math
import datetime
from binance.spot import Spot
from RMQTool import Tools as RMTTools
import os
import zipfile


def binance_to_his_csv(start_date, client, quot, remainder, k_interval):
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
        result_df.to_csv(path + "bar_crypto_BTCUSDT_d.csv")
        print("日线数据写入完成")
    elif k_interval == '4h':
        result_df.to_csv(path + "bar_crypto_BTCUSDT_240.csv")
        print("4h数据写入完成")
    elif k_interval == '1h':
        result_df.to_csv(path + "bar_crypto_BTCUSDT_60.csv")
        print("1h数据写入完成")
    elif k_interval == '15m':
        result_df.to_csv(path + "bar_crypto_BTCUSDT_15.csv")
        print("15m数据写入完成")


def get_history_crypto():
    start_date = datetime.datetime(2012, 4, 16)
    end_date = datetime.datetime(2024, 4, 16)
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
    binance_to_his_csv(start_date, client, dic['1d']['quot'], dic['1d']['remainder'], '1d')
    binance_to_his_csv(start_date, client, dic['4h']['quot'], dic['4h']['remainder'], '4h')
    binance_to_his_csv(start_date, client, dic['1h']['quot'], dic['1h']['remainder'], '1h')
    binance_to_his_csv(start_date, client, dic['15m']['quot'], dic['15m']['remainder'], '15m')


def process_crypto_data_current_month(code, time_level):
    # 3. 设置路径
    folder_path = f'D:/github/binance-public-data/python/data/spot/daily/klines/{code}/{time_level}/'

    all_data = []  # 用于存储当前月数据

    # 遍历该币种路径下的所有 zip 文件
    for zip_file in os.listdir(folder_path):
        if zip_file.endswith('.zip'):
            zip_file_path = os.path.join(folder_path, zip_file)

            # 解压并读取 CSV 文件
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # 假设 ZIP 文件中只有一个 CSV 文件
                csv_file = zip_ref.namelist()[0]
                with zip_ref.open(csv_file) as file:
                    df = pd.read_csv(file, header=None)

                    # 手动指定列名
                    df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                  'Quote asset volume',
                                  'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                                  'Ignore']

                    # 4. 只保留 Open、High、Low、Close、Volume、Close time 列
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Close time']]
                    # 5. 将 Close time 列从毫秒时间戳转换为 %Y-%m-%d %H:%M:%S 格式
                    # 注意，2025年元旦后，时间戳多了3位，从毫秒变成微秒了
                    df['Close time'] = df['Close time'].apply(lambda x: x // 1000 if len(str(int(x))) > 13 else x)
                    df['Close time'] = pd.to_datetime(df['Close time'] + 1, unit='ms')
                    df['Close time'] = df['Close time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # 6. 将列名转换为小写并修改 'close_time' 为 'time'
                    df.columns = ['open', 'high', 'low', 'close', 'volume', 'time']

                    # 7. 将 'time' 列移到首列
                    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

                    # 8. 将每个 CSV 文件的数据添加到列表中
                    all_data.append(df)
    merged_data = pd.concat(all_data, ignore_index=True)
    return merged_data


# 2. 定义一个函数来处理单个币种的数据
def process_crypto_data(code, time_level, target_time_level):
    # 3. 设置路径
    folder_path = f'D:/github/binance-public-data/python/data/spot/monthly/klines/{code}/{time_level}/'

    all_data = []  # 用于存储所有月度数据

    # 遍历该币种路径下的所有 zip 文件
    for zip_file in os.listdir(folder_path):
        if zip_file.endswith('.zip'):
            zip_file_path = os.path.join(folder_path, zip_file)

            # 解压并读取 CSV 文件
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # 假设 ZIP 文件中只有一个 CSV 文件
                csv_file = zip_ref.namelist()[0]
                with zip_ref.open(csv_file) as file:
                    df = pd.read_csv(file, header=None)

                    # 手动指定列名
                    df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                  'Quote asset volume',
                                  'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume',
                                  'Ignore']

                    # 4. 只保留 Open、High、Low、Close、Volume、Close time 列
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Close time']]
                    # 5. 将 Close time 列从毫秒时间戳转换为 %Y-%m-%d %H:%M:%S 格式
                    # 注意，2025年元旦后，时间戳多了3位，从毫秒变成微秒了
                    df['Close time'] = df['Close time'].apply(lambda x: x // 1000 if len(str(int(x))) > 13 else x)
                    df['Close time'] = pd.to_datetime(df['Close time'] + 1, unit='ms')
                    df['Close time'] = df['Close time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # 6. 将列名转换为小写并修改 'close_time' 为 'time'
                    df.columns = ['open', 'high', 'low', 'close', 'volume', 'time']

                    # 7. 将 'time' 列移到首列
                    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

                    # 8. 将每个 CSV 文件的数据添加到列表中
                    all_data.append(df)

    # 合并当前月的数据
    merged_data_day = process_crypto_data_current_month(code, time_level)
    # 9. 合并所有月度数据并按时间排序
    merged_data_month = pd.concat(all_data, ignore_index=True)
    merged_data = pd.concat([merged_data_month, merged_data_day], axis=0, ignore_index=True)
    merged_data = merged_data.sort_values(by='time').reset_index(drop=True)

    # 11. 保存最终结果到 CSV 文件
    output_file = f'D:/github/RobotMeQ/QuantData/backTest/bar_crypto_{code}_{target_time_level}.csv'
    merged_data.to_csv(output_file, index=False)
    print(f"已保存 {code} 的数据到 {output_file}")


if __name__ == '__main__':
    """
    获取历史数据
        历史数据官网是一条一条存储的，需要下载，再拼接
        我想那就不直接用接口调，1000条一拼接，从2020年4.21到2023年4.21
        每个级别都试，15m、1h、4h、day 再30m 2h、8h、  再5m、6h
        输入时间范围，级别；算出总共多少秒，算出有多少个对应的级别，换算成时间段，最后能自动拼接好数据
    """
    # get_history_crypto(start_date, end_date)

    """
    当初写这个代码时，还没有chatgpt，我也看不懂币安官网的数据格式，如今才搞明白。
    下载数字币数据要clone币安官网库，https://github.com/binance/binance-public-data/tree/master
    然后cd到python目录，我看了看文档，去执行python download-kline.py -t spot -s BTCUSDT -i 1d -startDate 2017-01-01
    但发现
        该命令旨在下载从 2017 年 1 月 1 日起的 BTCUSDT 日线级别（1d）K线数据。
        然而，您发现脚本同时下载了每个月的月度数据（每月一个 CSV 文件）和每日的数据（每天一个 CSV 文件），导致重复下载。
        这是因为脚本默认会下载月度和每日数据。要避免重复下载，您可以使用 -skip-monthly 或 -skip-daily 参数来跳过不需要的数据下载。
    所以应该执行
        python download-kline.py -t spot -s ETHUSDT -i 1d -startDate 2017-01-01 -skip-daily 1
        由于当前是 2025 年 2 月，月度数据仅更新到 2025 年 1 月，2 月的数据尚未完整，因此月度数据无法提供 2025 年 2 月的数据。
        所以执行完上面的，还应该配合执行下面这个
        python download-kline.py -t spot -s ETHUSDT -i 1d -startDate 2025-02-01 -skip-monthly 1

    我实验决定市值前9，热门新币上市时间太短，代码为：
        BTCUSDT ETHUSDT XRPUSDT SOLUSDT BNBUSDT DOGEUSDT ADAUSDT TRXUSDT LTCUSDT
        
    下载其他级别
        python download-kline.py -t spot -s BTCUSDT -i 4h -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 4h -startDate 2025-02-01 -skip-monthly 1
        
        python download-kline.py -t spot -s BTCUSDT -i 1h -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 1h -startDate 2025-02-01 -skip-monthly 1
        
        python download-kline.py -t spot -s BTCUSDT -i 15m -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 15m -startDate 2025-02-01 -skip-monthly 1
    """
    # 1. 创建一个包含数字币名称的 DataFrame
    codes = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LTCUSDT']
    df_codes = pd.DataFrame(codes, columns=['code'])
    # 11. 遍历所有数字币的 DataFrame 并处理
    for code in df_codes['code']:
        # 1d 4h 1h 15m   d 240 60 15
        process_crypto_data(code, "15m", "15")

