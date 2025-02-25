import os
import fnmatch
import pandas as pd

folder_path = "./QuantData/trade_point_backtest_c4_reversal_nature/"

# 遍历文件夹，获取所有符合 'USA_*_d.csv' 规则的文件
csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, '*_label1.csv')]

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # 检查是否包含 'label' 列
    if 'label' in df.columns:
        if df['label'].isnull().any():
            print(f"File: {file} - Contains NaN values in 'label' column")
            df = df.dropna(subset=['label'])  # 删除 label 列中包含 NaN 的行
            df.to_csv(file_path, index=False)  # 重新写入 CSV 文件
    else:
        print(f"File: {file} - No 'label' column found")