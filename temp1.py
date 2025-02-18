import os
import pandas as pd

# 过滤已经800里未处理的数据
# 文件夹路径和目标文件路径
# folder_path = "./QuantData/trade_point_backtest_fuzzy_nature/"  # 替换为存储CSV文件的文件夹路径
# hs_file_path = "./QuantData/a800_stocks.csv"  # hs.csv 文件路径
# output_file_path = "./QuantData/a800_wait_handle_stocks.csv"  # 输出文件路径
#
# # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# cc_list = [f[2:8] for f in file_names if len(f) >= 8]  # 提取第18到23位
# cc_unique = set(cc_list)  # 去重，转为集合方便快速查找
# # 步骤 2: 读取 hs.csv 文件并截取 code 列前3位后进行比较
# hs_df = pd.read_csv(hs_file_path)  # 假设 hs.csv 文件有多列
# hs_df['code_trimmed'] = hs_df['code'].astype(str).str[3:]  # 截取 code 列的第4位到末尾
# filtered_df = hs_df[~hs_df['code_trimmed'].isin(cc_unique)]  # 筛选出截取后不在 cc 中的行
#
# # 步骤 3: 删除临时列并保存结果到 a800_wait_handle_stocks.csv 文件
# filtered_df = filtered_df.drop(columns=['code_trimmed'])  # 删除辅助列
# filtered_df.to_csv(output_file_path, index=False)
#
# print(f"过滤完成，结果已保存到 {output_file_path}")

import os
import pandas as pd

# 文件夹路径
folder_path = "./QuantData/market_condition_backtest/"


# 统计函数：计算 market_condition 中连续 >= 5 行的情况
def count_consecutive_sequences(series, target_condition, min_length=5):
    count = 0
    streak = 0

    for value in series:
        if value == target_condition:
            streak += 1
        else:
            if streak >= min_length:
                count += 1
            streak = 0  # 重新计数

    # 处理最后一段连续的情况
    if streak >= min_length:
        count += 1

    return count


# 结果存储
summary = []

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith("_60.csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 计算连续出现 >= 5 次的情况
        trend_down_count = count_consecutive_sequences(df["market_condition"], "trend_down")
        trend_up_count = count_consecutive_sequences(df["market_condition"], "trend_up")
        range_count = count_consecutive_sequences(df["market_condition"], "range")

        # 记录结果
        summary.append([filename, trend_down_count, trend_up_count, range_count])

# 转换为 DataFrame 方便查看
result_df = pd.DataFrame(summary, columns=["File", "Trend Down (≥5)", "Trend Up (≥5)", "Range (≥5)"])

# 输出统计结果
print(result_df)

# 可选：保存到 CSV 文件
#result_df.to_csv("market_condition_consecutive_summary.csv", index=False)
