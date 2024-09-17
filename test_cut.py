import pandas as pd

# 读取CSV文件
df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

# 截掉最后24行数据
df_trimmed = df[:-24]

# 将剩余的数据保存为tt2.csv文件
df_trimmed.to_csv('D:/github/dataset/backtest_bar_600332_30_pred24.csv')
