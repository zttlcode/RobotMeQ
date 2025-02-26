import pandas as pd
import os
import zipfile


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
                    if time_level == '1d':
                        df['Close time'] = df['Close time'].dt.strftime('%Y-%m-%d')
                    else:
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
                    if time_level == '1d':
                        df['Close time'] = df['Close time'].dt.strftime('%Y-%m-%d')
                    else:
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
    纪念代码 如何使用代理
    proxies = {
        'http': 'http://127.0.0.1:33210',
        'https': 'http://127.0.0.1:33210',
    }
    client = Spot(proxies=proxies)
    """
    """
    当初写这个代码时(2025 0224 删了旧代码)，还没有chatgpt，我也看不懂币安官网的数据格式，如今才搞明白。
    下载数字币数据要clone币安官网库，https://github.com/binance/binance-public-data/tree/master
    然后cd到python目录，我看了看文档，去执行
        python download-kline.py -t spot -s BTCUSDT -i 1d -startDate 2017-01-01 
        该命令旨在下载从 2017 年 1 月 1 日起的 BTCUSDT 日线级别（1d）K线数据。同时下载了每个月的月度数据（每月一个 CSV 文件） 
        和每日的数据（每天一个 CSV 文件），导致重复下载。要避免重复下载，您可以使用 -skip-monthly 或 -skip-daily 参数来跳过不需要的数据下载。
        由于当前是 2025 年 2 月，月度数据仅更新到 2025 年 1 月，2 月的数据尚未完整，因此月度数据无法提供 2025 年 2 月的数据。
        所以执行完按月获取，还应该执行按日获取，正确命令如下，我获取4个级别的历史行情数据：
        
        python download-kline.py -t spot -s BTCUSDT -i 1d -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 1d -startDate 2025-02-01 -skip-monthly 1
        
        python download-kline.py -t spot -s BTCUSDT -i 4h -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 4h -startDate 2025-02-01 -skip-monthly 1
        
        python download-kline.py -t spot -s BTCUSDT -i 1h -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 1h -startDate 2025-02-01 -skip-monthly 1
        
        python download-kline.py -t spot -s BTCUSDT -i 15m -startDate 2017-01-01 -skip-daily 1
        python download-kline.py -t spot -s BTCUSDT -i 15m -startDate 2025-02-01 -skip-monthly 1
        
    我实验决定市值前9，热门新币上市时间太短，代码为：
        BTCUSDT ETHUSDT XRPUSDT SOLUSDT BNBUSDT DOGEUSDT ADAUSDT TRXUSDT LTCUSDT
    替换上面代码名即可
    下载完成后，要解压文件并组合成我需要的格式
    """
    # 1. 创建一个包含数字币名称的 DataFrame
    codes = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LTCUSDT']
    df_codes = pd.DataFrame(codes, columns=['code'])
    # 11. 遍历所有数字币的 DataFrame 并处理
    for code in df_codes['code']:
        # process_crypto_data(code, "15m", "15")
        # process_crypto_data(code, "1h", "60")
        # process_crypto_data(code, "4h", "240")
        process_crypto_data(code, "1d", "d")

