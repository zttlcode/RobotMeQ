# import os
# import glob
#
# # 设置文件夹路径
# folder_path = './QuantData/trade_point_backTest/'  # 替换为你实际的文件夹路径
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
# folder_path = './QuantData/trade_point_backTest/'  # 替换为实际的文件夹路径
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