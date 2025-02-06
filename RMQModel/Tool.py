import os
import glob
import pandas as pd


def process_fuzzy_trade_point_csv():
    def process_csv(file_path):
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 记录初始状态
        modified = False

        # 删除第一行如果label列不是'buy'
        if df.iloc[0]['signal'] != 'buy':
            df = df.iloc[1:]
            modified = True

        # 删除最后一行如果label列不是'sell'
        if df.iloc[-1]['signal'] != 'sell':
            df = df.iloc[:-1]
            modified = True

        # 仅在数据被修改时才写入CSV
        if modified:
            df.to_csv(file_path, index=False)
            print(f"文件 {file_path} 已修改并保存！")

    def process_folder(folder_path):
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        # 遍历所有CSV文件并处理
        for file in csv_files:
            process_csv(file)

    # 使用示例
    folder_path = '../QuantData/trade_point_backtest_fuzzy_nature/'  # 替换为你的文件夹路径
    process_folder(folder_path)


def count_label_distribution():
    folder_path = "../QuantData/trade_point_backtest_fuzzy_nature/"
    """ 遍历目标文件夹，统计每个CSV文件中 label 列的分布情况 """
    # 获取所有以 label1 结尾的 CSV 文件
    csv_files = glob.glob(os.path.join(folder_path, "*label1.csv"))

    for file in csv_files:
        try:
            df = pd.read_csv(file)  # 读取 CSV 文件

            if 'label' in df.columns:  # 确保有 label 列
                # 统计 label 列中 1、2、3、4 各自的数量
                label_counts = df['label'].value_counts().reindex([1, 2, 3, 4], fill_value=0)

                # 打印结果
                print(f"文件: {os.path.basename(file)}")
                print(f"  Label 1: {label_counts[1]} 行")
                print(f"  Label 2: {label_counts[2]} 行")
                print(f"  Label 3: {label_counts[3]} 行")
                print(f"  Label 4: {label_counts[4]} 行")
                print("-" * 40)

            else:
                print(f"文件 {file} 缺少 label 列，跳过处理。")

        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")


if __name__ == '__main__':
    # process_fuzzy_trade_point_csv()
    # 执行统计
    count_label_distribution()
