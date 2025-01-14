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


import akshare as ak

# 上证指数分钟级数据找到， 然后组装训练集
#
# 港股，美股，历史行情只有日线，没有分钟级，交易点位太少。
# 就算换策略也一样， 一个股票最多50个点位，800个股票才4万行
# 我分钟级一个股票500个点位，800个股票40万行训练集
index_zh_a_hist_min_em_df = ak.index_zh_a_hist_min_em(symbol="000001", period="5")
print(index_zh_a_hist_min_em_df)