import pandas as pd
import numpy as np
import os
from RMQTool import Tools as RMTTools
import RMQData.Indicator as RMQIndicator


def tea_radical_nature_label1(assetList, strategy_name):
    """
    对数收益计算：
        对 buy 和 sell 信号，分别计算第20天、第30天和第40天的对数收益。
        统计对数收益的正负值数量：
        如果正数收益占比达到 2/3 或更多，则标记为正收益；
        否则标记为负收益。
    更新标记逻辑：
        对于 buy：
            如果收益率大于 10%，标记为 1；
            如果收益率未达标，根据 20 天、30 天、40 天对数收益的正负一致性，决定是否标记为 1 或 2。
        对于 sell：
            如果收益率小于 -10%，标记为 3；
            如果收益率未达标，根据 20 天、30 天、40 天对数收益的正负一致性，决定是否标记为 3 或 4。
    时间范围校验：
        在数据不足 40 天的情况下，将剩余信号标记为 0 并删除。

    改动：
        目前的标注方式刷掉了一般交易点，但策略导致的买入点还是太多  代表性的如下
            Label=1: 260 行
            Label=2: 253 行
            Label=3: 66 行
            Label=4: 50 行
        因此严格条件：20、30、40全部为正
            Label=1: 198 行
            Label=2: 315 行
            Label=3: 66 行
            Label=4: 50 行
        因此严格条件：10、20、30、40全部为正
            Label=1: 170 行
            Label=2: 343 行
            Label=3: 66 行
            Label=4: 50 行
        因此严格条件：5、10、20、30、40全部为正
            Label=1: 151 行
            Label=2: 362 行
            Label=3: 66 行
            Label=4: 50 行
    """
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + assetList[0].assetsMarket
                            + "_"
                            + assetList[0].assetsCode
                            + '_d.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + assetList[0].assetsMarket
                          + "_"
                          + assetList[0].assetsCode + "_concat" + ".csv")
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期（忽略时间部分）
        signal_date = signal_time.date()

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index.date:
            backtest_row = backtest_df.loc[backtest_df.index.date == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_days = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_days = next_40_days.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_days["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_day = next_40_days.iloc[4]["close"]
                            close_10_day = next_40_days.iloc[9]["close"]
                            close_20_day = next_40_days.iloc[19]["close"]
                            close_30_day = next_40_days.iloc[29]["close"]
                            close_40_day = next_40_days.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_day / trade_price),
                                np.log(close_10_day / trade_price),
                                np.log(close_20_day / trade_price),
                                np.log(close_30_day / trade_price),
                                np.log(close_40_day / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_days["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_day = next_40_days.iloc[19]["close"]
                            close_30_day = next_40_days.iloc[29]["close"]
                            close_40_day = next_40_days.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_day / trade_price),
                                np.log(close_30_day / trade_price),
                                np.log(close_40_day / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + assetList[0].assetsMarket
                      + "_"
                      + assetList[0].assetsCode + "_concat_label1.csv"))

    print(assetList[0].assetsCode + "标注完成")


def tea_radical_nature_label2(asset, strategy_name):
    """ """
    """
    单级别各自标记
    """
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_bars = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_bars = next_40_bars.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_bars["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_bar = next_40_bars.iloc[4]["close"]
                            close_10_bar = next_40_bars.iloc[9]["close"]
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_bar / trade_price),
                                np.log(close_10_bar / trade_price),
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_bars["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label2.csv"))


def tea_radical_nature_label3(asset, strategy_name):
    """ """
    """
    MACD指标标记
    """
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")

    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 60 行）  为了算指标，前面也要顺40行
                if backtest_position + 60 > len(backtest_df):
                    # 如果剩余数据不足 60 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                elif backtest_position <= 40:
                    # 太靠前了，过
                    continue
                else:
                    # 获取从当前行开始顺延 40 行数据
                    window_100_bar = backtest_df.iloc[backtest_position - 40: backtest_position + 60].reset_index(
                        drop=True)
                    window_100_bar = RMQIndicator.calMACD(window_100_bar)
                    trade_MACD = window_100_bar.iloc[40]["MACD"]
                    trade_DIF = window_100_bar.iloc[40]["DIF"]
                    trade_DEA = window_100_bar.iloc[40]["DEA"]
                    offset = 100.0  # 为了解决对数计算有负数，增加偏移量

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 计算第20天、第30天和第40天的对数收益
                        MACD_5_bar = window_100_bar.iloc[44]["MACD"]
                        MACD_10_bar = window_100_bar.iloc[49]["MACD"]

                        DIF_5_bar = window_100_bar.iloc[44]["DIF"]
                        DIF_10_bar = window_100_bar.iloc[49]["DIF"]

                        DEA_5_bar = window_100_bar.iloc[44]["DEA"]
                        DEA_10_bar = window_100_bar.iloc[49]["DEA"]

                        log_returns = [
                            np.log((MACD_5_bar + offset) / (trade_MACD + offset)),
                            np.log((MACD_10_bar + offset) / (trade_MACD + offset)),
                            np.log((DIF_5_bar + offset) / (trade_DIF + offset)),
                            np.log((DIF_10_bar + offset) / (trade_DIF + offset)),
                            np.log((DEA_5_bar + offset) / (trade_DEA + offset)),
                            np.log((DEA_10_bar + offset) / (trade_DEA + offset)),
                        ]

                        # 判断对数收益的正负一致性
                        positive_count = sum(lr >= 0 for lr in log_returns)
                        negative_count = sum(lr < 0 for lr in log_returns)

                        if positive_count == 6:
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 计算第20天、第30天和第40天的对数收益
                        MACD_10_bar = window_100_bar.iloc[49]["MACD"]

                        DIF_10_bar = window_100_bar.iloc[49]["DIF"]

                        log_returns = [
                            np.log((MACD_10_bar + offset) / (trade_MACD + offset)),
                            np.log((DIF_10_bar + offset) / (trade_DIF + offset)),
                        ]

                        # 判断对数收益的正负一致性
                        positive_count = sum(lr >= 0 for lr in log_returns)
                        negative_count = sum(lr < 0 for lr in log_returns)

                        if negative_count == 2:
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label3.csv"))


def tea_radical_nature_label4(asset, strategy_name):
    """ """
    """
    MACD指标标记 + 价格过滤
    """
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 60 行）  为了算指标，前面也要顺40行
                if backtest_position + 60 > len(backtest_df):
                    # 如果剩余数据不足 60 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                elif backtest_position <= 40:
                    # 太靠前了，过
                    continue
                else:
                    # 获取从当前行开始顺延 40 行数据
                    window_100_bar = backtest_df.iloc[backtest_position - 40: backtest_position + 60].reset_index(
                        drop=True)
                    window_100_bar = RMQIndicator.calMACD(window_100_bar)
                    trade_MACD = window_100_bar.iloc[40]["MACD"]
                    trade_DIF = window_100_bar.iloc[40]["DIF"]
                    trade_DEA = window_100_bar.iloc[40]["DEA"]
                    trade_price = signal_row["price"]
                    offset = 100.0  # 为了解决对数计算有负数，增加偏移量

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 计算第20天、第30天和第40天的对数收益
                        MACD_5_bar = window_100_bar.iloc[44]["MACD"]
                        MACD_10_bar = window_100_bar.iloc[49]["MACD"]

                        DIF_5_bar = window_100_bar.iloc[44]["DIF"]
                        DIF_10_bar = window_100_bar.iloc[49]["DIF"]

                        DEA_5_bar = window_100_bar.iloc[44]["DEA"]
                        DEA_10_bar = window_100_bar.iloc[49]["DEA"]

                        # 计算第20天、第30天和第40天的对数收益
                        close_5_bar = window_100_bar.iloc[44]["close"]
                        close_10_bar = window_100_bar.iloc[49]["close"]
                        close_20_bar = window_100_bar.iloc[59]["close"]
                        close_30_bar = window_100_bar.iloc[69]["close"]
                        close_40_bar = window_100_bar.iloc[79]["close"]

                        log_returns = [
                            np.log((MACD_5_bar + offset) / (trade_MACD + offset)),
                            np.log((MACD_10_bar + offset) / (trade_MACD + offset)),
                            np.log((DIF_5_bar + offset) / (trade_DIF + offset)),
                            np.log((DIF_10_bar + offset) / (trade_DIF + offset)),
                            np.log((DEA_5_bar + offset) / (trade_DEA + offset)),
                            np.log((DEA_10_bar + offset) / (trade_DEA + offset)),
                            np.log(close_5_bar / trade_price),
                            np.log(close_10_bar / trade_price),
                            np.log(close_20_bar / trade_price),
                            np.log(close_30_bar / trade_price),
                            np.log(close_40_bar / trade_price),
                        ]

                        # 判断对数收益的正负一致性
                        positive_count = sum(lr >= 0 for lr in log_returns)
                        negative_count = sum(lr < 0 for lr in log_returns)

                        if positive_count == 11:
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 计算第20天、第30天和第40天的对数收益
                        MACD_10_bar = window_100_bar.iloc[49]["MACD"]

                        DIF_10_bar = window_100_bar.iloc[49]["DIF"]

                        # 计算第20天、第30天和第40天的对数收益
                        close_20_bar = window_100_bar.iloc[59]["close"]
                        close_30_bar = window_100_bar.iloc[69]["close"]
                        close_40_bar = window_100_bar.iloc[79]["close"]

                        log_returns = [
                            np.log((MACD_10_bar + offset) / (trade_MACD + offset)),
                            np.log((DIF_10_bar + offset) / (trade_DIF + offset)),
                            np.log(close_20_bar / trade_price),
                            np.log(close_30_bar / trade_price),
                            np.log(close_40_bar / trade_price),
                        ]

                        # 判断对数收益的正负一致性
                        positive_count = sum(lr >= 0 for lr in log_returns)
                        negative_count = sum(lr < 0 for lr in log_returns)

                        if negative_count >= 4:
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label4.csv"))


def fuzzy_nature_label1(asset, strategy_name):
    """ """
    """
    单级别各自标记
    """
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 signal.csv 为 DataFrame
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")

    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历信号点，查找 buy-sell 配对
    buy_price = None
    buy_index = None

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_index, signal_row in signal_df.iterrows():
        signal = signal_row["signal"]
        price = signal_row["price"]

        if signal == "buy":
            buy_price = price
            buy_index = signal_index  # 记录买入索引

        elif signal == "sell" and buy_price is not None:
            sell_price = price
            profit_ratio = (sell_price - buy_price) / buy_price  # 计算收益率

            if profit_ratio > 0.10:  # 盈利超过5%
                signal_df.at[buy_index, "label"] = 1  # 有效买入点
                signal_df.at[signal_index, "label"] = 3  # 有效卖出点
            else:
                signal_df.at[buy_index, "label"] = 2  # 无效买入点
                signal_df.at[signal_index, "label"] = 4  # 无效卖出点

            # 重置买入信息，等待下一个 buy-sell 组合
            buy_price = None
            buy_index = None

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def market_condition_label1(asset, strategy_name):
    """
    时间窗口，震荡5天，趋势12天 这算有效数据
    在label里把回测数据的连续5震荡挑出，连续12趋势挑出前5，分类为1趋势向上，2趋势向下，3震荡

    步长10，滑5次   这是逻辑上的滑，实际上把5行数据都放在label文件了，dataset会找每行数据的前step行组特征
        在dataset里找到每个日期的前5天，组特征。再试前5天
    """
    item = 'market_condition_backtest'
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    df = pd.read_csv(signal_df_filepath)

    time_label_records = []  # 记录满足条件的 time 和 label

    streak = 0
    current_condition = None
    start_index = None  # 记录连续段的起点

    for i in range(len(df)):
        condition = df.loc[i, "market_condition"]

        if condition == current_condition:
            streak += 1
        else:
            # 处理上一个连续段
            if current_condition is not None:
                if (current_condition == "trend_up" and streak > 12) or \
                        (current_condition == "trend_down" and streak > 12) or \
                        (current_condition == "range" and streak > 5):

                    # 计算 label
                    label = 1 if current_condition == "trend_up" else \
                        2 if current_condition == "trend_down" else \
                            3

                    # 取连续段前5行的 time
                    for j in range(min(5, streak)):
                        time_label_records.append([df.loc[start_index + j, "time"], label])

            # 重新初始化 streak 计算
            current_condition = condition
            streak = 1
            start_index = i  # 记录新连续段的起点

    # 处理最后一个连续段（避免遗漏）
    if current_condition is not None and (
            (current_condition == "trend_up" and streak > 12) or
            (current_condition == "trend_down" and streak > 12) or
            (current_condition == "range" and streak > 5)
    ):
        label = 1 if current_condition == "trend_up" else \
            2 if current_condition == "trend_down" else \
                3
        for j in range(min(5, streak)):
            time_label_records.append([df.loc[start_index + j, "time"], label])

    # 转换为 DataFrame 并保存
    result_df = pd.DataFrame(time_label_records, columns=["time", "label"])

    # 保存结果到新的 CSV 文件
    result_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"), index=False)


def c4_trend_nature_label1(asset, strategy_name):
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_bars = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_bars = next_40_bars.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_bars["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_bar = next_40_bars.iloc[4]["close"]
                            close_10_bar = next_40_bars.iloc[9]["close"]
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_bar / trade_price),
                                np.log(close_10_bar / trade_price),
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_bars["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def c4_oscillation_boll_nature_label1(asset, strategy_name):
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_bars = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_bars = next_40_bars.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_bars["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_bar = next_40_bars.iloc[4]["close"]
                            close_10_bar = next_40_bars.iloc[9]["close"]
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_bar / trade_price),
                                np.log(close_10_bar / trade_price),
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_bars["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def c4_oscillation_kdj_nature_label1(asset, strategy_name):
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 signal.csv 为 DataFrame
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")

    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历信号点，查找 buy-sell 配对
    buy_price = None
    buy_index = None

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_index, signal_row in signal_df.iterrows():
        signal = signal_row["signal"]
        price = signal_row["price"]

        if signal == "buy":
            buy_price = price
            buy_index = signal_index  # 记录买入索引

        elif signal == "sell" and buy_price is not None:
            sell_price = price
            profit_ratio = (sell_price - buy_price) / buy_price  # 计算收益率

            if profit_ratio > 0.05:  # 盈利超过5%
                signal_df.at[buy_index, "label"] = 1  # 有效买入点
                signal_df.at[signal_index, "label"] = 3  # 有效卖出点
            else:
                signal_df.at[buy_index, "label"] = 2  # 无效买入点
                signal_df.at[signal_index, "label"] = 4  # 无效卖出点

            # 重置买入信息，等待下一个 buy-sell 组合
            buy_price = None
            buy_index = None

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def c4_breakout_nature_label1(asset, strategy_name):
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_bars = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_bars = next_40_bars.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_bars["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_bar = next_40_bars.iloc[4]["close"]
                            close_10_bar = next_40_bars.iloc[9]["close"]
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_bar / trade_price),
                                np.log(close_10_bar / trade_price),
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_bars["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def c4_reversal_nature_label1(asset, strategy_name):
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                            + "bar_"
                            + asset.assetsMarket
                            + "_"
                            + asset.assetsCode
                            + "_"
                            + asset.barEntity.timeLevel
                            + '.csv')
    item = 'trade_point_backtest_' + strategy_name
    signal_df_filepath = (RMTTools.read_config("RMQData", item)
                          + asset.assetsMarket
                          + "_"
                          + asset.assetsCode
                          + "_"
                          + asset.barEntity.timeLevel
                          + ".csv")
    if not os.path.exists(signal_df_filepath):
        return None
    # 读取 backtest.csv读取日线数据 和 signal.csv 为 DataFrame
    backtest_df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")
    signal_df = pd.read_csv(signal_df_filepath, parse_dates=["time"], index_col="time")
    # 创建一个新列 'label'，用于标注信号数据
    signal_df["label"] = np.nan

    # 遍历 signal_df，按时间对比 backtest_df
    for signal_time, signal_row in signal_df.iterrows():
        # 获取 signal 的日期
        if asset.barEntity.timeLevel == 'd':
            # （忽略时间部分）
            signal_date = signal_time.strftime("%Y-%m-%d")
        else:
            signal_date = signal_time

        # 在 backtest_df 中找到对应日期的行
        if signal_date in backtest_df.index:
            backtest_row = backtest_df.loc[backtest_df.index == signal_date]

            if not backtest_row.empty:
                # 获取 backtest_df 中这一天的索引
                backtest_index = backtest_row.index[0]
                # 获取整数索引位置
                backtest_position = backtest_df.index.get_loc(backtest_index)

                # 检查是否有足够的剩余数据（顺延 40 行）
                if backtest_position + 40 > len(backtest_df):
                    # 如果剩余数据不足 40 行，将剩余数据标记为 0
                    signal_df.loc[signal_time:, "label"] = 0
                    break  # 直接退出循环
                else:
                    # 获取从当前行开始顺延 40 行数据
                    next_40_bars = backtest_df.iloc[backtest_position: backtest_position + 40]
                    next_20_bars = next_40_bars.iloc[:20]
                    trade_price = signal_row["price"]

                    # 根据信号类型执行不同的逻辑
                    if signal_row["signal"] == "buy":
                        # 获取 close 列的最大值
                        max_close = next_20_bars["close"].max()
                        return_rate = (max_close / trade_price) - 1

                        if return_rate > 0.1:  # 收益率是否大于10%
                            signal_df.at[signal_time, "label"] = 1
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_5_bar = next_40_bars.iloc[4]["close"]
                            close_10_bar = next_40_bars.iloc[9]["close"]
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_5_bar / trade_price),
                                np.log(close_10_bar / trade_price),
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if positive_count == 5:
                                signal_df.at[signal_time, "label"] = 1
                            else:
                                signal_df.at[signal_time, "label"] = 2

                    elif signal_row["signal"] == "sell":
                        # 获取 close 列的最小值
                        min_close = next_20_bars["close"].min()
                        return_rate = (min_close / trade_price) - 1

                        if return_rate < -0.1:  # 收益率是否小于-10%
                            signal_df.at[signal_time, "label"] = 3
                        else:
                            # 计算第20天、第30天和第40天的对数收益
                            close_20_bar = next_40_bars.iloc[19]["close"]
                            close_30_bar = next_40_bars.iloc[29]["close"]
                            close_40_bar = next_40_bars.iloc[39]["close"]

                            log_returns = [
                                np.log(close_20_bar / trade_price),
                                np.log(close_30_bar / trade_price),
                                np.log(close_40_bar / trade_price),
                            ]

                            # 判断对数收益的正负一致性
                            positive_count = sum(lr > 0 for lr in log_returns)
                            negative_count = sum(lr <= 0 for lr in log_returns)

                            if negative_count >= 2:
                                signal_df.at[signal_time, "label"] = 3
                            else:
                                signal_df.at[signal_time, "label"] = 4

    # 删除标记为 0 的数据
    signal_df = signal_df[signal_df["label"] != 0]

    # 保存结果到新的 CSV 文件
    signal_df.to_csv((RMTTools.read_config("RMQData", item)
                      + asset.assetsMarket
                      + "_"
                      + asset.assetsCode
                      + "_"
                      + asset.barEntity.timeLevel
                      + "_label1.csv"))


def label(assetList, strategy_name, label_name):
    if label_name == "label1":
        if strategy_name == "tea_radical_nature":
            tea_radical_nature_label1(assetList, strategy_name)
        elif strategy_name == "fuzzy_nature":
            for asset in assetList:
                fuzzy_nature_label1(asset, strategy_name)
        elif strategy_name == "extremum":
            for asset in assetList:
                fuzzy_nature_label1(asset, strategy_name)
        elif strategy_name == "identify_Market_Types":
            for asset in assetList:
                market_condition_label1(asset, strategy_name)
        elif strategy_name == "c4_trend_nature":
            for asset in assetList:
                c4_trend_nature_label1(asset, strategy_name)
        elif strategy_name == "c4_oscillation_boll_nature":
            for asset in assetList:
                c4_oscillation_boll_nature_label1(asset, strategy_name)
        elif strategy_name == "c4_oscillation_kdj_nature":
            for asset in assetList:
                c4_oscillation_kdj_nature_label1(asset, strategy_name)
        elif strategy_name == "c4_breakout_nature":
            for asset in assetList:
                c4_breakout_nature_label1(asset, strategy_name)
        elif strategy_name == "c4_reversal_nature":
            for asset in assetList:
                c4_reversal_nature_label1(asset, strategy_name)
    elif label_name == "label2":
        for asset in assetList:
            tea_radical_nature_label2(asset, strategy_name)
    elif label_name == "label3":
        for asset in assetList:
            tea_radical_nature_label3(asset, strategy_name)
    elif label_name == "label4":
        for asset in assetList:
            tea_radical_nature_label4(asset, strategy_name)

    print(assetList[0].assetsCode + "标注完成")
