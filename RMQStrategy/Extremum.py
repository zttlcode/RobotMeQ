import pandas as pd
import os
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def plot_waves(prices, wave_points, wave_prices):
    """绘制波浪结构，包括 ABC 调整浪"""
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", linewidth=1.5)
    plt.scatter(wave_points, wave_prices, color='red', label="Wave Points", zorder=3)

    # 标注浪型编号
    for i, (x, y) in enumerate(zip(wave_points, wave_prices)):
        plt.text(x, y, f"{i + 1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Elliott Wave")
    plt.show()


def detect_waves(data_1, distance=5, prominence=1):
    """检测价格的局部峰值（高点）和谷值（低点），并返回包含time、price和signal的DataFrame"""
    prices = data_1["close"].to_numpy()

    # 检测局部峰值（高点）和谷值（低点）
    peaks, _ = signal.find_peaks(prices, distance=distance, prominence=prominence)  # 高点
    valleys, _ = signal.find_peaks(-prices, distance=distance, prominence=prominence)  # 低点
    try:
        # 获取峰谷点对应的时间和价格
        wave_times = data_1['time'].iloc[np.concatenate([peaks, valleys])].values
        wave_prices = prices[np.concatenate([peaks, valleys])]

        # 创建包含time, price和signal的DataFrame
        wave_df = pd.DataFrame({
            'time': wave_times,
            'price': wave_prices,
            'signal': ['sell' if i in peaks else 'buy' for i in np.concatenate([peaks, valleys])]

        })
        # 按时间排序
        wave_df = wave_df.sort_values(by='time')

        # 确保信号的正确顺序
        if wave_df['signal'].iloc[0] != 'buy':
            wave_df = wave_df.iloc[1:]  # 删除第一个点
        if wave_df['signal'].iloc[-1] != 'sell':
            wave_df = wave_df.iloc[:-1]  # 删除最后一个点

        # 假设 wave_df 是你的 DataFrame
        # 通过 shift() 来比较当前行和前一行的值
        wave_df = wave_df[wave_df['signal'] != wave_df['signal'].shift(-1)]
        wave_df = wave_df.reset_index(drop=True)

        # # 可视化波浪
        # wave_points = sorted(np.concatenate([peaks, valleys]))
        # wave_prices = prices[wave_points]
        # plot_waves(prices, wave_points, wave_prices)
    except Exception as e:
        print(e)
        return pd.DataFrame()

    return wave_df


def preprocess_stock_data(allStockCode):
    """处理所有股票数据并保存最终结果
    """
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:], row['code_name'], ['d'], 'stock', 1, 'A')

        for asset in assetList:
            data_1_filePath = (RMTTools.read_config("RMQData", "backtest_bar") +
                               "bar_" + asset.assetsMarket + "_" + asset.assetsCode +
                               "_" + asset.barEntity.timeLevel + ".csv")
            data_1 = pd.read_csv(data_1_filePath, parse_dates=["time"])

            # 调用detect_waves，得到包含time, price, signal的DataFrame
            wave_df = detect_waves(data_1)

            if wave_df.empty:
                continue
            # 保存结果为csv文件
            item = 'trade_point_backtest_' + "extremum"
            directory = RMTTools.read_config("RMQData", item)
            os.makedirs(directory, exist_ok=True)
            wave_df.to_csv(directory
                           + asset.assetsMarket
                           + "_"
                           + asset.assetsCode
                           + "_"
                           + asset.barEntity.timeLevel
                           + ".csv", index=False)
            print(asset.assetsCode, "结束")


if __name__ == '__main__':
    # 运行处理函数 传统极值标注法回测数据
    allStockCode = pd.read_csv("../QuantData/a800_stocks.csv", dtype={'code': str})
    # df_dataset = allStockCode.iloc[500:]
    # 执行完此函数，要执行process_fuzzy_trade_point_csv()，再用fuzzy_nature_label1标注，否则会导致模型训练时label分类中含有nan，模型训练报错
    # preprocess_stock_data(allStockCode)
