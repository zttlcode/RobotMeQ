# import os
# import glob
# import pandas as pd
#
# def find_zero_close_files(folder_path):
#     # 获取所有CSV文件
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
#
#     # 遍历CSV文件
#     for file in csv_files:
#         try:
#             df = pd.read_csv(file)
#
#             # 检查是否存在 'close' 列，并且是否有值为0
#             if 'close' in df.columns and (df['close'] == 0).any():
#                 print(f"文件包含 close=0: {file}")
#
#         except Exception as e:
#             print(f"读取文件 {file} 时发生错误: {e}")
#
#
# # 使用示例
# folder_path = "./QuantData/backTest/"  # 请替换为实际的文件夹路径
# find_zero_close_files(folder_path)
#
#

#
# import os
# import glob
# import pandas as pd
#
# def extract_codes_from_csv(folder_path):
#     """
#     遍历文件夹中的所有 CSV 文件，找到 close=0 的文件，并提取文件名的 6-12 位字符。
#     """
#     extracted_codes = set()  # 使用集合存储提取的代码，避免重复
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))  # 获取所有 CSV 文件
#
#     for file in csv_files:
#         try:
#             df = pd.read_csv(file)  # 读取 CSV 文件
#
#             # 检查是否存在 'close' 列，并查找 close=0 的行
#             if 'close' in df.columns and (df['close'] == 0).any():
#                 filename = os.path.basename(file)  # 获取文件名（不含路径）
#                 extracted_code = filename[6:12]   # 提取 6-12 位字符
#                 extracted_codes.add(extracted_code)
#
#         except Exception as e:
#             print(f"读取文件 {file} 时发生错误: {e}")
#
#     return extracted_codes
#
# def remove_matching_codes(a800_csv_path, extracted_codes):
#     """
#     读取 a800_stocks.csv 文件，删除 code 列中去掉前三位后匹配 extracted_codes 的行，并保存。
#     """
#     try:
#         df = pd.read_csv(a800_csv_path)  # 读取 a800_stocks.csv
#
#         if 'code' in df.columns:
#             # 提取后三位（截掉前三位）
#             df["trimmed_code"] = df["code"].astype(str).str[3:]
#
#             # 过滤出不在 extracted_codes 集合中的行
#             df_filtered = df[~df["trimmed_code"].isin(extracted_codes)]
#
#             # 删除辅助列 trimmed_code
#             df_filtered = df_filtered.drop(columns=["trimmed_code"])
#
#             # 仅在有变动时才保存
#             if len(df) != len(df_filtered):
#                 df_filtered.to_csv(a800_csv_path, index=False)
#                 print(f"已删除匹配的行，并更新文件: {a800_csv_path}")
#             else:
#                 print("没有匹配的行，无需修改 a800_stocks.csv")
#
#     except Exception as e:
#         print(f"处理文件 {a800_csv_path} 时发生错误: {e}")
#
# # 使用示例
# folder_path = "./QuantData/backTest/"  # 替换为 CSV 文件所在的文件夹
# a800_csv_path = "./QuantData/a800_stocks.csv"  # a800_stocks.csv 文件路径
#
# # 提取 close=0 的文件名 6-12 位字符
# codes_to_remove = extract_codes_from_csv(folder_path)
#
# # 根据提取的代码删除 a800_stocks.csv 中匹配的行
# remove_matching_codes(a800_csv_path, codes_to_remove)


#
# import os
# import glob
# import pandas as pd
#
# # 设置路径
# a800_csv_path = "./QuantData/a800_stocks.csv"
# backtest_folder = "./QuantData/trade_point_backtest_fuzzy_nature/"
#
# def get_valid_codes(a800_csv_path):
#     """ 从 a800_stocks.csv 提取去掉前三位的 code 列，形成有效代码集合 """
#     try:
#         df = pd.read_csv(a800_csv_path)
#         if 'code' in df.columns:
#             valid_codes = set(df["code"].astype(str).str[3:])  # 截断前三位
#             return valid_codes
#         else:
#             print(f"文件 {a800_csv_path} 不包含 'code' 列")
#             return set()
#     except Exception as e:
#         print(f"读取文件 {a800_csv_path} 失败: {e}")
#         return set()
#
# def delete_unmatched_files(backtest_folder, valid_codes):
#     """ 遍历 backTest 文件夹，删除文件名 6-12 位与 valid_codes 不匹配的文件 """
#     backtest_files = glob.glob(os.path.join(backtest_folder, "*"))  # 获取所有文件
#
#     for file in backtest_files:
#         filename = os.path.basename(file)  # 获取文件名（去掉路径）
#         extracted_code = filename[2:8]   # 提取 6-12 位字符
#
#         if extracted_code not in valid_codes:  # 如果不匹配
#             try:
#                 os.remove(file)  # 删除文件
#                 print(f"已删除: {file}")
#             except Exception as e:
#                 print(f"删除 {file} 失败: {e}")
#
# # 执行
# valid_codes = get_valid_codes(a800_csv_path)
# delete_unmatched_files(backtest_folder, valid_codes)



