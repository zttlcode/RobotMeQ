import pandas as pd
from RMQTool import Tools as RMTTools
import RMQData.Indicator as RMQIndicator

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
