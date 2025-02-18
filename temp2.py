# import os
# from collections import Counter
#
# # 文件夹路径
# folder_path = "QuantData/trade_point_backtest_fuzzy_nature/"  # 替换为你的文件夹路径
#
# # 步骤 1: 获取所有文件名，并提取 cc 值
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# cc_list = [f[2:8] for f in file_names]  # 提取文件名第18到23位
#
# # 步骤 2: 统计每个 cc 的出现次数
# cc_counter = Counter(cc_list)
#
# # 步骤 3: 找出出现次数不是 5 的 cc
# unexpected_cc = {cc: count for cc, count in cc_counter.items() if count != 3}
#
# # 步骤 4: 输出结果
# if unexpected_cc:
#     print("以下 cc 值出现次数异常：")
#     for cc, count in unexpected_cc.items():
#         print(f"cc: {cc}, count: {count}")
# else:
#     print("所有 cc 值的出现次数均为 5")
#
# """
# 没d  因为单边一直跌，导致MACD区域少于3个
# cc: 000408, count: 4
# cc: 001289, count: 4
# cc: 301269, count: 4
# cc: 301498, count: 4
# cc: 600745, count: 4
# cc: 600918, count: 4
# cc: 600941, count: 4
# cc: 601061, count: 4
# cc: 601136, count: 4
# cc: 601728, count: 4
# cc: 603296, count: 4
# cc: 603341, count: 4
# cc: 688072, count: 4
# cc: 688349, count: 4
# cc: 688361, count: 4
# cc: 688469, count: 4
# """

import os
import pandas as pd

# 文件夹路径
folder_path = "./QuantData/trade_point_backtest_c4_trend_nature/"
output_file = "./QuantData/trade_point_backtest_c4_trend_nature/signal_count_summary.csv"  # 结果文件

# 用于存储每个 CSV 文件的统计结果
all_results = []

# 遍历所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取 CSV
        df = pd.read_csv(file_path, index_col=False)

        # 统计 signal 列的值分布
        signal_counts = df["signal"].value_counts().to_dict()

        # 将结果转换为 DataFrame 格式，方便后续合并
        result = {"filename": filename}
        result.update(signal_counts)  # 把统计结果添加到字典
        all_results.append(result)  # 存入结果列表

# 汇总所有结果
summary_df = pd.DataFrame(all_results).fillna(0)  # NaN 填充为 0

# 保存到 CSV
summary_df.to_csv(output_file, index=False)

print(f"统计完成，结果已保存到 {output_file}")

