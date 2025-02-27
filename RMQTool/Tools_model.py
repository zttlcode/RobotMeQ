import os
import glob
import pandas as pd
from collections import Counter
import fnmatch
import re


def find_zero_close_files():
    # 使用示例
    folder_path = "../QuantData/backTest/"  # 请替换为实际的文件夹路径
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 遍历CSV文件
    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # 检查是否存在 'close' 列，并且是否有值为0
            if 'close' in df.columns and (df['close'] == 0).any():
                print(f"文件包含 close=0: {file}")

        except Exception as e:
            print(f"读取文件 {file} 时发生错误: {e}")


# A股股票数据中有close为0的，直接从a800里删除，更新a800文件
def process_backtest_nan_data_a800():
    def extract_codes_from_csv(folder_path):
        """
        遍历文件夹中的所有 CSV 文件，找到 close=0 的文件，并提取文件名的 6-12 位字符。
        """
        extracted_codes = set()  # 使用集合存储提取的代码，避免重复
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))  # 获取所有 CSV 文件

        for file in csv_files:
            try:
                df = pd.read_csv(file)  # 读取 CSV 文件

                # 检查是否存在 'close' 列，并查找 close=0 的行
                if 'close' in df.columns and (df['close'] == 0).any():
                    filename = os.path.basename(file)  # 获取文件名（不含路径）
                    extracted_code = filename[7:12]  # 提取 6-12 位字符
                    extracted_codes.add(extracted_code)

            except Exception as e:
                print(f"读取文件 {file} 时发生错误: {e}")

        return extracted_codes

    def remove_matching_codes(a800_csv_path, extracted_codes):
        """
        读取 a800_stocks.csv 文件，删除 code 列中去掉前三位后匹配 extracted_codes 的行，并保存。
        """
        try:
            df = pd.read_csv(a800_csv_path)  # 读取 a800_stocks.csv

            if 'code' in df.columns:
                # 提取后三位（截掉前三位）
                df["trimmed_code"] = df["code"].astype(str).str[3:]

                # 过滤出不在 extracted_codes 集合中的行
                df_filtered = df[~df["trimmed_code"].isin(extracted_codes)]

                # 删除辅助列 trimmed_code
                df_filtered = df_filtered.drop(columns=["trimmed_code"])

                # 仅在有变动时才保存
                if len(df) != len(df_filtered):
                    df_filtered.to_csv(a800_csv_path, index=False)
                    print(f"已删除匹配的行，并更新文件: {a800_csv_path}")
                else:
                    print("没有匹配的行，无需修改 a800_stocks.csv")

        except Exception as e:
            print(f"处理文件 {a800_csv_path} 时发生错误: {e}")

    # 使用示例
    folder_path = "../QuantData/backTest/"  # 替换为 CSV 文件所在的文件夹
    a800_csv_path = "../QuantData/a800_stocks.csv"  # a800_stocks.csv 文件路径

    # 提取 close=0 的文件名 6-12 位字符
    codes_to_remove = extract_codes_from_csv(folder_path)

    # 根据提取的代码删除 a800_stocks.csv 中匹配的行
    remove_matching_codes(a800_csv_path, codes_to_remove)


# 之前没发现，所以把其他文件夹里含这股票的文件也删除
def process_backtest_nan_data_a800_for_other():
    # 设置路径
    a800_csv_path = "../QuantData/a800_stocks.csv"
    backtest_folder = "../QuantData/trade_point_backtest_fuzzy_nature/"

    def get_valid_codes(a800_csv_path):
        """ 从 a800_stocks.csv 提取去掉前三位的 code 列，形成有效代码集合 """
        try:
            df = pd.read_csv(a800_csv_path)
            if 'code' in df.columns:
                valid_codes = set(df["code"].astype(str).str[3:])  # 截断前三位
                return valid_codes
            else:
                print(f"文件 {a800_csv_path} 不包含 'code' 列")
                return set()
        except Exception as e:
            print(f"读取文件 {a800_csv_path} 失败: {e}")
            return set()

    def delete_unmatched_files(backtest_folder, valid_codes):
        """ 遍历 backTest 文件夹，删除文件名 6-12 位与 valid_codes 不匹配的文件 """
        backtest_files = glob.glob(os.path.join(backtest_folder, "*"))  # 获取所有文件

        for file in backtest_files:
            filename = os.path.basename(file)  # 获取文件名（去掉路径）
            extracted_code = filename[2:8]  # 提取 6-12 位字符

            if extracted_code not in valid_codes:  # 如果不匹配
                try:
                    os.remove(file)  # 删除文件
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除 {file} 失败: {e}")

    # 执行
    valid_codes = get_valid_codes(a800_csv_path)
    delete_unmatched_files(backtest_folder, valid_codes)


def process_fuzzy_trade_point_csv():
    def process_csv(file_path):
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 记录初始状态
        modified = False
        try:
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
        except Exception as e:
            print(file_path, e)

    def process_folder(folder_path):
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        # 遍历所有CSV文件并处理
        for file in csv_files:
            process_csv(file)

    # 使用示例
    folder_path = '../QuantData/trade_point_backtest_c4_oscillation_kdj_nature/'  # 替换为你的文件夹路径
    process_folder(folder_path)


def handle_800_wait():
    # 过滤已经800里未处理的数据
    # 文件夹路径和目标文件路径
    folder_path = "../QuantData/market_condition_backtest/"  # 替换为存储CSV文件的文件夹路径
    hs_file_path = "../QuantData/asset_code/a800_stocks.csv"  # hs.csv 文件路径
    output_file_path = "../QuantData/asset_code/a800_stocks_wait_handle_stocks.csv"  # 输出文件路径

    # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
    file_names = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'A_*_60.csv')]
    cc_list = [f[2:8] for f in file_names if len(f) >= 8]  # 提取第18到23位
    cc_unique = set(cc_list)  # 去重，转为集合方便快速查找
    # 步骤 2: 读取 hs.csv 文件并截取 code 列前3位后进行比较
    hs_df = pd.read_csv(hs_file_path)  # 假设 hs.csv 文件有多列
    hs_df['code_trimmed'] = hs_df['code'].astype(str).str[3:]  # 截取 code 列的第4位到末尾
    filtered_df = hs_df[~hs_df['code_trimmed'].isin(cc_unique)]  # 筛选出截取后不在 cc 中的行

    # 步骤 3: 删除临时列并保存结果到 a800_stocks_wait_handle_stocks.csv 文件
    filtered_df = filtered_df.drop(columns=['code_trimmed'])  # 删除辅助列
    filtered_df.to_csv(output_file_path, index=False)

    print(f"过滤完成，结果已保存到 {output_file_path}")


def handle_hk_1000_wait():
    # 过滤已经800里未处理的数据
    # 文件夹路径和目标文件路径
    folder_path = "../QuantData/market_condition_backtest/"  # 替换为存储CSV文件的文件夹路径
    hs_file_path = "../QuantData/hk_1000_stock_codes.csv"  # hs.csv 文件路径
    output_file_path = "../QuantData/hk_1000_stock_codes_wait_handle_stocks.csv"  # 输出文件路径

    # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
    file_names = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'HK_*_d.csv')]
    cc_list = [f[3:8] for f in file_names if len(f) >= 7]  # 提取第18到23位
    cc_unique = set(cc_list)  # 去重，转为集合方便快速查找
    # 步骤 2: 读取 hs.csv 文件并截取 code 列前3位后进行比较
    hs_df = pd.read_csv(hs_file_path, dtype={'code': str})  # 假设 hs.csv 文件有多列
    hs_df['code_trimmed'] = hs_df['code'].astype(str)  # 取 code 列
    filtered_df = hs_df[~hs_df['code_trimmed'].isin(cc_unique)]  # 筛选出截取后不在 cc 中的行

    # 步骤 3: 删除临时列并保存结果到 a800_stocks_wait_handle_stocks.csv 文件
    filtered_df = filtered_df.drop(columns=['code_trimmed'])  # 删除辅助列
    filtered_df.to_csv(output_file_path, index=False)

    print(f"过滤完成，结果已保存到 {output_file_path}")


def handle_sp500_wait():
    # 过滤已经800里未处理的数据
    # 文件夹路径和目标文件路径
    folder_path = "../QuantData/trade_point_backtest_c4_reversal_nature/"  # 替换为存储CSV文件的文件夹路径
    hs_file_path = "../QuantData/asset_code/sp500_stock_codes.csv"  # hs.csv 文件路径
    output_file_path = "../QuantData/asset_code/sp500_stock_codes_wait_handle_stocks.csv"  # 输出文件路径

    # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
    file_names = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'USA_*_d.csv')]
    cc_list = [re.search(r'USA_(.*?)_d.csv', f).group(1) for f in file_names if re.search(r'USA_(.*?)_d.csv', f)]

    cc_unique = set(cc_list)  # 去重，转为集合方便快速查找
    # 步骤 2: 读取 hs.csv 文件并截取 code 列前3位后进行比较
    hs_df = pd.read_csv(hs_file_path)  # 假设 hs.csv 文件有多列
    hs_df['code_trimmed'] = hs_df['code'].astype(str)  # 截取 code 列的第4位到末尾
    filtered_df = hs_df[~hs_df['code_trimmed'].isin(cc_unique)]  # 筛选出截取后不在 cc 中的行

    # 步骤 3: 删除临时列并保存结果到 a800_stocks_wait_handle_stocks.csv 文件
    filtered_df = filtered_df.drop(columns=['code_trimmed'])  # 删除辅助列
    filtered_df.to_csv(output_file_path, index=False)

    print(f"过滤完成，结果已保存到 {output_file_path}")


def handle_crypto_wait():
    # 过滤已经800里未处理的数据
    # 文件夹路径和目标文件路径
    folder_path = "../QuantData/trade_point_backtest_fuzzy_nature/"  # 替换为存储CSV文件的文件夹路径
    hs_file_path = "../QuantData/crypto_code.csv"  # hs.csv 文件路径
    output_file_path = "../QuantData/crypto_code_wait_handle_stocks.csv"  # 输出文件路径

    # 步骤 1: 获取文件夹中所有文件名，并提取 cc 列
    file_names = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'crypto_*_d.csv')]
    cc_list = [re.search(r'crypto_(.*?)_d.csv', f).group(1) for f in file_names if re.search(r'crypto_(.*?)_d.csv', f)]
    cc_unique = set(cc_list)  # 去重，转为集合方便快速查找
    # 步骤 2: 读取 hs.csv 文件并截取 code 列前3位后进行比较
    hs_df = pd.read_csv(hs_file_path)  # 假设 hs.csv 文件有多列
    hs_df['code_trimmed'] = hs_df['code'].astype(str)  # 截取 code 列的第4位到末尾
    filtered_df = hs_df[~hs_df['code_trimmed'].isin(cc_unique)]  # 筛选出截取后不在 cc 中的行

    # 步骤 3: 删除临时列并保存结果到 a800_stocks_wait_handle_stocks.csv 文件
    filtered_df = filtered_df.drop(columns=['code_trimmed'])  # 删除辅助列
    filtered_df.to_csv(output_file_path, index=False)

    print(f"过滤完成，结果已保存到 {output_file_path}")


def count_consecutive_sequences(series, target_condition, min_length=10):
    # 统计函数：计算 market_condition 中连续 >= 5 行的情况
    count = 0
    streak = 0

    for value in series:
        if value == target_condition:
            streak += 1
        else:
            if streak >= min_length:
                count += 1
            streak = 0  # 重新计数

    # 处理最后一段连续的情况
    if streak >= min_length:
        count += 1

    return count


def count_market_condition_for_label():
    # 标注前，先看各类分布情况，再决定要不要处理样本不均
    # 文件夹路径
    folder_path = "../QuantData/market_condition_backtest/"

    # 结果存储
    summary = []

    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if fnmatch.fnmatch(filename, 'crypto_*_240.csv'):
            file_path = os.path.join(folder_path, filename)

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 计算连续出现 >= 5 次的情况
            trend_down_count = count_consecutive_sequences(df["market_condition"], "trend_down")
            trend_up_count = count_consecutive_sequences(df["market_condition"], "trend_up")
            range_count = count_consecutive_sequences(df["market_condition"], "range")

            # 记录结果
            summary.append([filename, trend_down_count, trend_up_count, range_count])

    # 转换为 DataFrame 方便查看
    result_df = pd.DataFrame(summary, columns=["File", "Trend Down (≥5)", "Trend Up (≥5)", "Range (≥5)"])

    # 输出统计结果
    print(result_df)

    # 可选：保存到 CSV 文件
    # result_df.to_csv("market_condition_consecutive_summary.csv", index=False)


def count_market_condition_for_preprocessing():
    # 回测完行情分类，统计每个文件中，各分类占比多少

    # 文件夹路径
    folder_path = "../QuantData/market_condition_backtest/"

    # 遍历文件夹中的所有 CSV 文件
    summary = {}

    for filename in os.listdir(folder_path):
        if fnmatch.fnmatch(filename, 'crypto_*_d.csv'):
            file_path = os.path.join(folder_path, filename)

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 统计 market_condition 各取值的数量
            condition_counts = df["market_condition"].value_counts().to_dict()

            # 存入结果字典
            summary[filename] = condition_counts

    # 输出统计结果
    for file, counts in summary.items():
        print(f"File: {file}")
        for condition, count in counts.items():
            print(f"  {condition}: {count}")
        print()


def count_signal_before_label():
    # 文件夹路径
    folder_path = "../QuantData/trade_point_backtest_c4_trend_nature/"
    output_file = "../QuantData/trade_point_backtest_c4_trend_nature/signal_count_summary.csv"  # 结果文件

    # 用于存储每个 CSV 文件的统计结果
    all_results = []

    # 遍历所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith("d.csv"):
            file_path = os.path.join(folder_path, filename)

            # 读取 CSV
            df = pd.read_csv(file_path, index_col=False)

            # 统计 signal 列的值分布
            signal_counts = df["signal"].value_counts().to_dict()

            # 将结果转换为 DataFrame 格式，方便后续合并
            result = {"filename": filename}
            result.update(signal_counts)  # 把统计结果添加到字典
            all_results.append(result)  # 存入结果列表

    # 汇总所有结果
    summary_df = pd.DataFrame(all_results).fillna(0)  # NaN 填充为 0

    # 保存到 CSV
    summary_df.to_csv(output_file, index=False)

    print(f"统计完成，结果已保存到 {output_file}")


def count_trade_point_csv_files():
    """
    有些数据策略不出信号，所以这里统计一下有多少，不过也没啥用了，我后面都加了文件不存在校验
    回测时若没有这个文件，直接跳过
    没d  因为单边一直跌，导致MACD区域少于3个
    cc: 000408, count: 4
    cc: 001289, count: 4
    cc: 301269, count: 4
    cc: 301498, count: 4
    cc: 600745, count: 4
    cc: 600918, count: 4
    cc: 600941, count: 4
    cc: 601061, count: 4
    cc: 601136, count: 4
    cc: 601728, count: 4
    cc: 603296, count: 4
    cc: 603341, count: 4
    cc: 688072, count: 4
    cc: 688349, count: 4
    cc: 688361, count: 4
    cc: 688469, count: 4
    """
    # 文件夹路径
    folder_path = "QuantData/trade_point_backtest_fuzzy_nature/"  # 替换为你的文件夹路径

    # 步骤 1: 获取所有文件名，并提取 cc 值
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    cc_list = [f[2:8] for f in file_names]  # 提取文件名第18到23位

    # 步骤 2: 统计每个 cc 的出现次数
    cc_counter = Counter(cc_list)

    # 步骤 3: 找出出现次数不是 5 的 cc
    unexpected_cc = {cc: count for cc, count in cc_counter.items() if count != 3}

    # 步骤 4: 输出结果
    if unexpected_cc:
        print("以下 cc 值出现次数异常：")
        for cc, count in unexpected_cc.items():
            print(f"cc: {cc}, count: {count}")
    else:
        print("所有 cc 值的出现次数均为 5")


def count_label_distribution():
    folder_path = "../QuantData/trade_point_backtest_extremum/"
    """ 遍历目标文件夹，统计每个CSV文件中 label 列的分布情况 """
    csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'A_*_d_label1.csv')]

    # 获取所有以 label1 结尾的 CSV 文件
    # csv_files = glob.glob(os.path.join(folder_path, "*_d_label1.csv"))
    label_counts1 = 0
    label_counts2 = 0
    label_counts3 = 0
    label_counts4 = 0
    for file in csv_files:
        try:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)  # 读取 CSV 文件

            if 'label' in df.columns:  # 确保有 label 列
                # 统计 label 列中 1、2、3、4 各自的数量
                label_counts = df['label'].value_counts().reindex([1, 2, 3, 4], fill_value=0)

                # 打印结果
                # print(f"文件: {os.path.basename(file)}")
                # print(f"  Label 1: {label_counts[1]} 行")
                # print(f"  Label 2: {label_counts[2]} 行")
                # print(f"  Label 3: {label_counts[3]} 行")
                # print(f"  Label 4: {label_counts[4]} 行")
                # print("-" * 40)
                label_counts1 += label_counts[1]
                label_counts2 += label_counts[2]
                label_counts3 += label_counts[3]
                label_counts4 += label_counts[4]
            else:
                print(f"文件 {file} 缺少 label 列，跳过处理。")

        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")
    print(f"  Label 1: {label_counts1} 行")
    print(f"  Label 2: {label_counts2} 行")
    print(f"  Label 3: {label_counts3} 行")
    print(f"  Label 4: {label_counts4} 行")


def count_label_null():
    # 查看label列是否为nan，如果是，则删除那一行
    folder_path = "./QuantData/trade_point_backtest_tea_radical_nature/"

    # 遍历文件夹，获取所有符合 'USA_*_d.csv' 规则的文件
    csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, '*_label2.csv')]

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


def check_time_sort():
    # 查看时间列是否为升序，如果不是，中间数据乱了，这个原因估计是run里多个策略同时运行，导致订单文件重复读取

    folder_path = "./QuantData/trade_point_backtest_c4_reversal_nature/"

    # 遍历文件夹，获取所有符合 'USA_*_d.csv' 规则的文件
    # csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, 'HK_*_d.csv')]
    csv_files = [f for f in os.listdir(folder_path) if fnmatch.fnmatch(f, '*.csv')]

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # 检查是否包含 'date' 列
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')  # 转换为 datetime 类型，处理异常值
            if df['time'].isnull().any():
                print(f"File: {file} - Contains invalid date values")
                continue

            # 检查是否严格递增
            if not df['time'].is_monotonic_increasing:
                print(f"File: {file} - Date column is not strictly increasing")
        else:
            print(f"File: {file} - No 'date' column found")


if __name__ == '__main__':
    count_label_distribution()

