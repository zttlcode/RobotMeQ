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

import pandas as pd

# 初始化一个空的 DataFrame
df = pd.DataFrame()

# 模拟循环生成数据
for i in range(10):  # 假设循环10次
    # 假设每次循环生成的 6 个 Series 对象
    series1 = pd.Series([i * 1, i * 2, i * 3, i * 4, i * 5])
    series2 = pd.Series([i * 2, i * 4, i * 6, i * 8, i * 10])
    series3 = pd.Series([i * 3, i * 6, i * 9, i * 12, i * 15])
    series4 = pd.Series([i * 4, i * 8, i * 12, i * 16, i * 20])
    series5 = pd.Series([i * 5, i * 10, i * 15, i * 20, i * 25])
    series6 = pd.Series([i * 6, i * 12, i * 18, i * 24, i * 30])

    # 将这些 Series 组成一行
    new_row = pd.DataFrame({
        'col1': [series1.values],
        'col2': [series2.values],
        'col3': [series3.values],
        'col4': [series4.values],
        'col5': [series5.values],
        'col6': [series6.values]
    })

    # 追加到主 DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

# 查看结果
print(df)

