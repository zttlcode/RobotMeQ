import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RMQTool import Tools as RMTTools
import RMQData.Indicator as RMQIndicator


def preprocess_data_for_OT():
    # 给时序模型准备数据，OT列是预测的明天的股价
    filePath = RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_600332_30.csv'
    DataFrame = pd.read_csv(filePath, encoding='gbk', parse_dates=['time'])  # 假设'时间'列包含日期时间，所以将其解析为datetime类型
    DataFrame = RMQIndicator.calMACD(DataFrame)
    DataFrame = RMQIndicator.calKDJ(DataFrame)
    DataFrame = RMQIndicator.calMA(DataFrame)

    DataFrame['MA_5'].fillna(DataFrame['MA_5'].iloc[4], inplace=True)
    DataFrame['MA_10'].fillna(DataFrame['MA_10'].iloc[9], inplace=True)
    DataFrame['MA_60'].fillna(DataFrame['MA_60'].iloc[59], inplace=True)
    DataFrame = DataFrame.rename(columns={'time': 'date'})

    # 确保索引不是时间列（如果已经是，可以跳过这一步）
    DataFrame.set_index('date', inplace=True)
    # 创建一个新的列OT，并初始化为NaN（对于除了最后一行以外的所有行）
    DataFrame['OT'] = pd.NaT  # 使用NaT来表示时间类型的缺失值，或者你也可以使用float的NaN
    # 对于除了最后一行以外的所有行，设置OT为下一行的收盘价
    for i in range(len(DataFrame) - 1):
        DataFrame.iloc[i, DataFrame.columns.get_loc('OT')] = DataFrame.iloc[i + 1, DataFrame.columns.get_loc('close')]
    # 最后一行的OT设置为NaN（或者其他默认值，如0）
    DataFrame.iloc[-1, DataFrame.columns.get_loc('OT')] = DataFrame.iloc[-1, DataFrame.columns.get_loc('close')]
    # 或者使用float的NaN，或者其他默认值
    # 如果需要，将索引重置回默认的整数索引
    DataFrame.reset_index(inplace=True)

    DataFrame.to_csv(filePath, index=False)
    # 如果需要，将结果保存回CSV文件
    DataFrame.to_csv('../dataset/backtest_bar_600332_30.csv', index=False)  # 不保存索引
    # DataFrame.to_csv('../dataset/index/backtest_bar_N_d.csv', index=False)  # 不保存索引
    # DataFrame.to_csv('../dataset/cryptocurrency/backtest_bar_BTCUSDT_60.csv', index=False)  # 不保存索引


def cut_data_for_OT_pred():
    # 读取CSV文件
    df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

    # 截掉最后24行数据
    df_trimmed = df[:-24]

    # 将剩余的数据保存为tt2.csv文件
    df_trimmed.to_csv('D:/github/dataset/backtest_bar_600332_30_pred24.csv')


def draw_data_for_OT_pred():
    # 读取CSV文件
    df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

    # 截取最后100行数据的'close'列
    df_Y = df['OT'].tail(500)

    # 绘制实际值的曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(df_Y.index, df_Y.values, color='red', label='Actual Close', linewidth=0.5)

    # 添加图例
    plt.legend()

    # 设置标题和坐标轴标签
    plt.title('Close Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Close Value')

    # 显示图形
    plt.show()


def compare_actual_pred_close():
    # 读取CSV文件
    df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

    # 截取最后100行数据的'close'列
    df_last_100 = df['OT'].tail(30)

    # 绘制实际值的曲线图
    plt.figure(figsize=(20, 6))
    plt.plot(df_last_100.index, df_last_100.values, color='red', label='Actual Close', linewidth=0.5)

    # 加载预测数据
    # data = np.load('D:/github/PatchTST/PatchTST_supervised/results/test_PatchTST_custom_ftMS_sl192_ll96_pl48_dm500_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
    # data = np.load('D:/github/Autoformer/results/test_Autoformer_custom_ftMS_sl192_ll96_pl48_dm512_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
    # data = np.load('D:/github/LTSF-Linear/results/test_DLinear_custom_ftMS_sl16_ll8_pl4_dm512_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
    data = np.load(
        'D:/github/PatchTST/PatchTST_supervised/results/test_PatchTST_custom_ftMS_sl96_ll48_pl24_dm500_nh20_el2_dl1_df32_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')

    # 使用切片保留后两维
    data_sliced = data[0, :, :]
    close_column = data_sliced[:, -1]
    print(close_column)
    close_column_reshaped = close_column.reshape(-1, 1)
    arr_raveled = close_column_reshaped.ravel()
    # 绘制预测值的曲线图
    plt.plot(df_last_100.index[-24:], arr_raveled, color='blue', label='Predicted Close')

    # 添加图例
    plt.legend()

    # 设置标题和坐标轴标签
    plt.title('Close Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Close Value')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # preprocess_data_for_OT()
    # cut_data_for_OT_pred()
    # compare_actual_pred_close()
    pass
