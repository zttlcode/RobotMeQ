# import os
# import pandas as pd
#
# # 过滤已经800里未处理的数据
# # 文件夹路径和目标文件路径
# folder_path = "./QuantData/trade_point_backtest_tea_radical/"  # 替换为存储CSV文件的文件夹路径
# hs_file_path = "./QuantData/a800_stocks.csv"  # hs.csv 文件路径
# output_file_path = "./QuantData/a800_wait_handle_stocks.csv"  # 输出文件路径
#
# # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# cc_list = [f[17:23] for f in file_names if len(f) >= 23]  # 提取第18到23位
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
