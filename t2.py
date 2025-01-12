import os
from collections import Counter

# 文件夹路径
folder_path = "./QuantData/trade_point_backTest/"  # 替换为你的文件夹路径

# 步骤 1: 获取所有文件名，并提取 cc 值
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
cc_list = [f[17:23] for f in file_names if len(f) >= 23]  # 提取文件名第18到23位

# 步骤 2: 统计每个 cc 的出现次数
cc_counter = Counter(cc_list)

# 步骤 3: 找出出现次数不是 5 的 cc
unexpected_cc = {cc: count for cc, count in cc_counter.items() if count != 5}

# 步骤 4: 输出结果
if unexpected_cc:
    print("以下 cc 值出现次数异常：")
    for cc, count in unexpected_cc.items():
        print(f"cc: {cc}, count: {count}")
else:
    print("所有 cc 值的出现次数均为 5")

"""
没d
cc: 000155, count: 4
cc: 000629, count: 4
cc: 002558, count: 4
cc: 301498, count: 4
cc: 600153, count: 4
cc: 600673, count: 4
cc: 601061, count: 4
cc: 688349, count: 4
cc: 688361, count: 4
cc: 688469, count: 4
"""