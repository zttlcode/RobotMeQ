import pandas as pd
import akshare as ak

import RMQData.Asset as RMQAsset
import Run

"""
akshare==1.16.9
需要python3.9及以上
"""

def get_sp500_data():
    # 读取s&p500_stock_codes.csv文件，获取股票代码
    df = pd.read_csv('../QuantData/asset_code/sp500_stock_codes.csv')
    stock_codes = df['code'].tolist()

    # 循环遍历每个股票代码
    for symbol in stock_codes:
        try:
            # 获取股票历史数据
            stock_data = ak.stock_us_daily(symbol=symbol, adjust="")  # 调整方式根据需要可以修改

            # 修改列名，将"date"改为"time"
            stock_data.rename(columns={'date': 'time'}, inplace=True)

            # 保存数据为CSV文件，文件名为股票代码
            stock_data.to_csv(f'D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_USA_{symbol}_d.csv', index=False)

            print(f"{symbol} 数据已保存为 {symbol}.csv")
        except Exception as e:
            print(f"无法获取 {symbol} 的数据: {e}")


def get_hk_stock_data():
    # 读取s&p500_stock_codes.csv文件，获取股票代码
    df = pd.read_csv('../QuantData/asset_code/hk_1000_stock_codes.csv', dtype={'code': str})

    stock_codes = df['code'].tolist()

    # 循环遍历每个股票代码
    for symbol in stock_codes:
        try:
            # 获取股票历史数据
            stock_data = ak.stock_hk_daily(symbol=symbol, adjust="")  # 调整方式根据需要可以修改

            # 修改列名，将"date"改为"time"
            stock_data.rename(columns={'date': 'time'}, inplace=True)

            # 保存数据为CSV文件，文件名为股票代码
            stock_data.to_csv(f'D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_HK_{symbol}_d.csv', index=False)

            print(f"{symbol} 数据已保存为 {symbol}.csv")
        except Exception as e:
            print(f"无法获取 {symbol} 的数据: {e}")


def get_hk_stock_code():
    # 获取港股知名股，110多家
    # stock_hk_famous_spot_em_df = ak.stock_hk_famous_spot_em()
    # stock_hk_famous_spot_em_df_filtered = stock_hk_famous_spot_em_df[['代码', '名称']]
    # stock_hk_famous_spot_em_df_filtered.to_csv('../QuantData/hk_famous.csv', index=False)

    # 获取港股所有股票
    # stock_hk_spot_em_df = ak.stock_hk_spot_em()
    # stock_hk_spot_em_df_filtered = stock_hk_spot_em_df[['代码', '名称']]
    # stock_hk_spot_em_df_filtered.to_csv('../QuantData/hk_all_stocks.csv', index=False)

    df1 = pd.read_csv('../QuantData/asset_code/hk_all_stocks.csv', dtype={'代码': str})
    df2 = pd.read_csv('../QuantData/asset_code/hk_famous.csv', dtype={'代码': str})

    # 获取df1的前1000行  这个不是按市值排的，我在网上找不到按市值排名的数据
    df1_top_1000 = df1.head(1000)

    # 合并df1和df2
    df_combined = pd.concat([df1_top_1000, df2], ignore_index=True)

    # 重命名列名
    df_combined.rename(columns={'代码': 'code', '名称': 'name'}, inplace=True)

    # 按code列去重
    df_combined_unique = df_combined.drop_duplicates(subset=['code'])

    # 保存为新的CSV文件
    df_combined_unique.to_csv('../QuantData/asset_code/hk_1000_stock_codes.csv', index=False)


if __name__ == '__main__':
    # 获取股票代码、获取股票发行日、获取股票各级别数据
    # get_stock_from_code_csv()  # 日线能从发行日开始，分钟级别最早是2019年元旦
    assetsCode = 'NVDA'  # NVDA TSLA 01810
    assetList = RMQAsset.asset_generator(assetsCode,
                                         assetsCode,
                                         ['d'],
                                         'stock',
                                         0, 'USA')  # USA HK
    df = ak.stock_us_daily(symbol=assetsCode, adjust="")  # 获取股票历史数据
    # df = ak.stock_hk_daily(symbol=assetsCode, adjust="")  # 获取股票历史数据
    # 20250909为了stock_hk_daily给run_live_A800_TSLA加了live_df.index = pd.to_datetime(live_df.index)代码

    df.rename(columns={'date': 'time'}, inplace=True)
    Run.run_live_A800_TSLA(assetList, df)