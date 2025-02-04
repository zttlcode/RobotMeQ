import os

# # 设置目标文件夹路径
# folder_path = './QuantData/backTest/'
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件名是否以 "trade_point_list_" 开头
#     if filename.startswith("backtest_bar_"):
#         # 修改文件名，将开头替换为 "A_"
#         new_filename = filename.replace("backtest_bar_", "bar_A_", 1)
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)
#
#         # 重命名文件
#         os.rename(old_path, new_path)
#         print(f"文件已重命名：{filename} -> {new_filename}")
import pandas as pd

# 读取数据
df_5 = pd.read_csv("./QuantData/backTest/bar_A_000001_5.csv", parse_dates=["time"])
df_15 = pd.read_csv("./QuantData/backTest/bar_A_000001_15.csv", parse_dates=["time"])

# 创建一个新的列存储行号
df_15["row_number"] = range(0, len(df_15))

# 遍历 df_5 的 time 列
for time_5 in df_5["time"]:
    # 计算15分钟K线的区间
    start_time = time_5.floor("15T")  # 向下取整到最近的15分钟
    end_time = start_time + pd.Timedelta(minutes=15)

    # 在 df_15 中查找符合区间的行
    mask = (df_15["time"] > start_time) & (df_15["time"] <= end_time)
    matching_indices = df_15.index[mask].to_numpy()  # 使用 to_numpy() 解决 FutureWarning

    if matching_indices.size > 0:
        prev_index = matching_indices[-1] - 1  # 获取上一行的索引

        # 检查是否越界
        if prev_index < 0:
            continue

        # 获取上一行的行号并打印
        prev_row_number = df_15.loc[prev_index, "time"]
        print(prev_row_number)


