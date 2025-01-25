# import os
# import pandas as pd
#
# # 过滤已经800里未处理的数据
# # 文件夹路径和目标文件路径
# folder_path = "./QuantData/trade_point_backtest_tea_radical_nature/"  # 替换为存储CSV文件的文件夹路径
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


# import os
# import glob
#
# # 设置文件夹路径
# folder_path = './QuantData/trade_point_backtest_tea_radical_nature/'  # 替换为你实际的文件夹路径
#
# # 使用 glob 模块查找所有以 concat_labeled.csv 结尾的文件
# files = glob.glob(os.path.join(folder_path, '*concat_labeled.csv'))
#
# # 获取文件数量
# file_count = len(files)
#
# # 打印结果
# print(f"文件夹下以 'concat_labeled.csv' 结尾的文件数量是: {file_count}")


# import os
# import glob
# import pandas as pd
#
# # 设置文件夹路径
# folder_path = './QuantData/trade_point_backtest_tea_radical_nature/'  # 替换为实际的文件夹路径
#
# # 查找所有以 concat_labeled.csv 结尾的文件
# files = glob.glob(os.path.join(folder_path, '*concat_labeled.csv'))
#
# # 遍历文件，统计每个文件中 label 列的分布
# for file in files:
#     # 读取文件为 DataFrame
#     df = pd.read_csv(file)
#
#     # 检查是否有 label 列
#     if 'label' in df.columns:
#         # 统计 label 列中每种取值的行数
#         label_counts = df['label'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
#
#         # 打印结果
#         print(f"文件名: {os.path.basename(file)}")
#         print(f"Label=1: {label_counts[1]} 行")
#         print(f"Label=2: {label_counts[2]} 行")
#         print(f"Label=3: {label_counts[3]} 行")
#         print(f"Label=4: {label_counts[4]} 行")
#         print("-" * 40)
#     else:
#         print(f"文件名: {os.path.basename(file)} 中未找到 'label' 列")
#         print("-" * 40)


# import akshare as ak
#
# # 上证指数分钟级数据找到， 然后组装训练集
# #
# # 港股，美股，历史行情只有日线，没有分钟级，交易点位太少。
# # 就算换策略也一样， 一个股票最多50个点位，800个股票才4万行
# # 我分钟级一个股票500个点位，800个股票40万行训练集
# index_zh_a_hist_min_em_df = ak.index_zh_a_hist_min_em(symbol="000001", period="5")
# print(index_zh_a_hist_min_em_df)

# import pandas as pd
#
# # 读取CSV文件
# file1 = pd.read_csv('file1.csv')  # 包含time列的数据
# file2 = pd.read_csv('file2.csv')  # 包含time列和close列的数据
# file3 = pd.read_csv('file3.csv')  # 包含time列和close列的数据
#
# # 创建一个字典来存储匹配的结果
# data_dict = {'close_file2': [], 'close_file3': []}
#
# # 遍历file1中的time列
# for time in file1['time']:
#     # 在file2和file3中找到time列相同的数据
#     data2 = file2[file2['time'] == time]
#     data3 = file3[file3['time'] == time]
#
#     # 如果找到匹配的数据
#     if not data2.empty and not data3.empty:
#         # 找到file2和file3中对应的close列的前5行数据
#         close_data2 = data2['close'].iloc[0:5].reset_index(drop=True)
#         close_data3 = data3['close'].iloc[0:5].reset_index(drop=True)
#
#         # 将结果添加到字典中
#         data_dict['close_file2'].append(close_data2)
#         data_dict['close_file3'].append(close_data3)
#
# # 将字典转换为DataFrame
# result = pd.DataFrame(data_dict)
#
# import pandas as pd
# import os
#
# # 指定文件夹路径
# folder_path = './QuantData/trade_point_backtest_tea_radical_nature/'
#
# # 获取文件夹下所有符合条件的 CSV 文件
# csv_files = [
#     f for f in os.listdir(folder_path)
#     if f.endswith('.csv') and not (f.endswith('_concat_labeled.csv') or f.endswith('_concat.csv'))
# ]
#
# for file in csv_files:
#     # 读取 CSV 文件
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)
#
#     # 检查是否有 3 列，修改列名
#     if df.shape[1] == 3:
#         df.columns = ['time', 'price', 'signal']
#
#         # 将修改后的 DataFrame 写回原始文件
#         df.to_csv(file_path, index=False)
#
#         print(f"file: {file}")
import pandas as pd

# 读取 temp.csv 和 temp2.csv
df1 = pd.read_csv('./QuantData/temp.csv')   # 假设 df1 有 1 列
df2 = pd.read_csv('./QuantData/temp2.csv')  # 假设 df2 的第一列是需要比较的列

# 将 df1 的列并入 df2 中
df2['df1_col'] = df1.iloc[:, 0]  # 添加 df1 的第一列到 df2，命名为 'df1_col'

# 比较 df1 列与 df2 第一列的值，并标记结果
df2['comparison'] = (df2['df1_col'] > df2.iloc[:, 0]).astype(int)  # 1 表示 df1 列值较大，0 表示较小或相等

# 统计标记列中 1 的总数
count_of_ones = df2['comparison'].sum()

# 将 df2 写回 temp2.csv
df2.to_csv('temp2.csv', index=False)

count_of_ones = df2['good'].sum()
# 输出标记列中 1 的总数
print(f"The number of 1s in the comparison column is: {count_of_ones}")