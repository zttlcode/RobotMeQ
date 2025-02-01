import os

# # 设置目标文件夹路径
# folder_path = './QuantData/backTest/'
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件名是否以 "trade_point_list_" 开头
#     if filename.startswith("backtest_bar_"):
#         # 修改文件名，将开头替换为 "A_"
#         new_filename = filename.replace("backtest_bar_", "bar_A_", 1)
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)
#
#         # 重命名文件
#         os.rename(old_path, new_path)
#         print(f"文件已重命名：{filename} -> {new_filename}")
import numpy as np
n1 = 100.02
n2 = 100.02
print(np.log(n1 / n2))
