import pandas as pd
import numpy as np

from RMQTool import Tools as RMTTools


def filter1(assetList):
    """ """
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
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_' +
                            assetList[0].assetsCode + '_d.csv')
    signal_df_filepath = (RMTTools.read_config("RMQData", "trade_point_backtest_tea_radical")
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
    signal_df.to_csv((RMTTools.read_config("RMQData", "trade_point_backtest_tea_radical")
                      + assetList[0].assetsMarket
                      + "_"
                      + assetList[0].assetsCode + "_concat_labeled" + ".csv"))

    print(assetList[0].assetsCode + "标注完成")
