from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from RMQTool import Tools as RMTTools
import RMQData.Asset as RMQAsset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# 斐波那契回调比率
FIB_RATIOS = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618])


def detect_waves(prices, distance=5, prominence=1):
    """检测价格的局部峰值（高点）和谷值（低点）"""
    peaks, _ = signal.find_peaks(prices, distance=distance, prominence=prominence)  # 高点
    valleys, _ = signal.find_peaks(-prices, distance=distance, prominence=prominence)  # 低点

    # 合并峰谷并按时间排序
    wave_points = sorted(np.concatenate([peaks, valleys]))
    wave_prices = prices[wave_points]

    return wave_points, wave_prices


def check_fibonacci(wave_prices):
    """检查波浪的斐波那契比例是否符合预期"""
    ratios = np.diff(wave_prices) / np.abs(np.diff(wave_prices)).max()
    return all(np.any(np.isclose(ratios[i] / ratios[i - 1], FIB_RATIOS, atol=0.05)) for i in range(1, len(ratios)))


def detect_adjustment_waves(wave_points, wave_prices):
    """识别 ABC 调整浪"""
    if len(wave_prices) < 5:
        return []  # 确保至少有 5 个浪型点

    abc_waves = []
    for i in range(len(wave_prices) - 4):
        a, b, c = wave_prices[i:i + 3]
        if a > b < c and (c - b) / (a - b) in FIB_RATIOS:
            abc_waves.append((wave_points[i], wave_points[i + 1], wave_points[i + 2]))

    return abc_waves


def plot_waves(prices, wave_points, wave_prices, abc_waves=[]):
    """绘制波浪结构，包括 ABC 调整浪"""
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", linewidth=1.5)
    plt.scatter(wave_points, wave_prices, color='red', label="Wave Points", zorder=3)

    # 标注浪型编号
    for i, (x, y) in enumerate(zip(wave_points, wave_prices)):
        plt.text(x, y, f"{i + 1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # 标注 ABC 调整浪
    for (a, b, c) in abc_waves:
        plt.plot([a, b, c], [prices[a], prices[b], prices[c]], 'bo-', label='ABC Pattern')

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Elliott Wave & ABC Correction Detection")
    plt.show()


allStockCode = pd.read_csv("QuantData/a800_stocks.csv")
for index, row in allStockCode.iterrows():
    assetList = RMQAsset.asset_generator(row['code'][3:],
                                         row['code_name'],
                                         ['30', '60', 'd'],
                                         'stock',
                                         1, 'A')
    data_1_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                       + "bar_"
                       + assetList[0].assetsMarket
                       + "_"
                       + assetList[0].assetsCode
                       + "_"
                       + assetList[0].barEntity.timeLevel
                       + ".csv")
    data_1 = pd.read_csv(data_1_filePath, parse_dates=["time"])
    prices = data_1["close"].to_numpy()

    # 识别波浪
    wave_points, wave_prices = detect_waves(prices)

    # 识别 ABC 调整浪
    abc_waves = detect_adjustment_waves(wave_points, wave_prices)

    # 检查斐波那契比例
    if check_fibonacci(wave_prices):
        print("符合波浪理论的结构！")
    else:
        print("不符合波浪理论，请调整参数或检查数据。")

    # 可视化波浪和 ABC 调整浪
    plot_waves(prices, wave_points, wave_prices, abc_waves)
    sleep(1)
