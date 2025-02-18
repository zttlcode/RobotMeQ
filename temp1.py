import os
import pandas as pd

#
# import os
# import pandas as pd
#
# # 文件夹路径
# folder_path = "./QuantData/market_condition_backtest/"
#
#
# # 统计函数：计算 market_condition 中连续 >= 5 行的情况
# def count_consecutive_sequences(series, target_condition, min_length=5):
#     count = 0
#     streak = 0
#
#     for value in series:
#         if value == target_condition:
#             streak += 1
#         else:
#             if streak >= min_length:
#                 count += 1
#             streak = 0  # 重新计数
#
#     # 处理最后一段连续的情况
#     if streak >= min_length:
#         count += 1
#
#     return count
#
#
# # 结果存储
# summary = []
#
# # 遍历文件夹中的所有 CSV 文件
# for filename in os.listdir(folder_path):
#     if filename.endswith("_60.csv"):
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取 CSV 文件
#         df = pd.read_csv(file_path)
#
#         # 计算连续出现 >= 5 次的情况
#         trend_down_count = count_consecutive_sequences(df["market_condition"], "trend_down")
#         trend_up_count = count_consecutive_sequences(df["market_condition"], "trend_up")
#         range_count = count_consecutive_sequences(df["market_condition"], "range")
#
#         # 记录结果
#         summary.append([filename, trend_down_count, trend_up_count, range_count])
#
# # 转换为 DataFrame 方便查看
# result_df = pd.DataFrame(summary, columns=["File", "Trend Down (≥5)", "Trend Up (≥5)", "Range (≥5)"])
#
# # 输出统计结果
# print(result_df)
#
# # 可选：保存到 CSV 文件
# #result_df.to_csv("market_condition_consecutive_summary.csv", index=False)
