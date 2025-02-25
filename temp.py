import pandas as pd
import matplotlib.pyplot as plt
#
# # 读取CSV文件
# df = pd.read_csv('D:/github/Time-Series-Library-Quant/data/b202212_d.csv', index_col='date', parse_dates=True)
#
# # 选择最后500行数据
# df_last_500 = df.tail(35000)
#
# # 选择除了'date'以外的所有列
# columns = df_last_500.columns
#
# # 创建一个图形，设置大小
# plt.figure(figsize=(12, 8))
#
# # 遍历每一列并绘制
# for column in columns:
#     plt.plot(df_last_500.index, df_last_500[column], label=column)
#
# # 添加标题和标签
# plt.title('Time Series of Each Variable (Last 500 Rows)')
# plt.xlabel('Date')
# plt.ylabel('Value')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()



# # 读取CSV文件
# df = pd.read_csv('D:/github/Time-Series-Library-Quant/data/b202212_d.csv', index_col='date', parse_dates=True)
#
# # 选择'MESFOC_nmile'列的最后500行数据
# df_last_500 = df['MESFOC_nmile'].head(5000)
#
# # 创建一个图形，设置大小
# plt.figure(figsize=(12, 8))
#
# # 绘制'MESFOC_nmile'列的数据
# plt.plot(df_last_500.index, df_last_500, label='MESFOC_nmile', color='b')
#
# # 添加标题和标签
# plt.title('MESFOC_nmile Time Series (Last 500 Rows)')
# plt.xlabel('Date')
# plt.ylabel('Value')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()


import os
import fnmatch
import pandas as pd

folder_path = "./QuantData/trade_point_backtest_c4_reversal_nature/"

# 遍历文件夹，获取所有符合 'USA_*_d.csv' 规则的文件
# csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'HK_*_d.csv')]
csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, '*.csv')]

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # 检查是否包含 'date' 列
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')  # 转换为 datetime 类型，处理异常值
        if df['time'].isnull().any():
            print(f"File: {file} - Contains invalid date values")
            continue

        # 检查是否严格递增
        if not df['time'].is_monotonic_increasing:
            print(f"File: {file} - Date column is not strictly increasing")
    else:
        print(f"File: {file} - No 'date' column found")
