"""
项目代号：本质
项目描述：炒股无非是看看各级别的k线、成交量，综合做决策，
本方法通过机器学习挖掘各级别之间的关系，分类当前交易点是否有效，并投票
实验用于回测，实盘用于实际交易

模型分类的，是此信号有效，无效，而不是买卖  有效为1，失效为0
有效就报信号，无效就不报
我的方法也抽取了某个时间窗口的特征，看现在的价格走势满足过去哪些特征，以此特征做识别，识别以前遇到这种情况指标会不会失效。  ！！！！！！！！

那我标注时，应该把所有交易点都留下，无效的标为无效，有效的标为有效，固定窗口
抓特征只抓上级

多种指标，就是多个特征，需要捕获特征之间的相关性么？像3论文大纲的 itransformer一样？
patchtst  分段？通道独立我可以试试，时间段token，每个特征各自进transformer，我模仿一下，每个特征各自进cnn？
"""
import pandas as pd
import numpy as np
import RMQStrategy.Strategy_nature as RMQStrategy
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools


def concat_trade_point(assetList):
    # 读取交易点
    tpl_filepath = RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_"
    df_tpl_5 = pd.read_csv(tpl_filepath +
                           assetList[0].assetsCode + "_" + assetList[0].barEntity.timeLevel + ".csv")
    df_tpl_15 = pd.read_csv(tpl_filepath +
                            assetList[1].assetsCode + "_" + assetList[1].barEntity.timeLevel + ".csv")
    df_tpl_30 = pd.read_csv(tpl_filepath +
                            assetList[2].assetsCode + "_" + assetList[2].barEntity.timeLevel + ".csv")
    df_tpl_60 = pd.read_csv(tpl_filepath +
                            assetList[3].assetsCode + "_" + assetList[3].barEntity.timeLevel + ".csv")
    df_tpl_d = None
    # temp2中有16个股票是单边行情，没用日线交易信号
    try:
        df_tpl_d = pd.read_csv(tpl_filepath +
                               assetList[4].assetsCode + "_" + assetList[4].barEntity.timeLevel + ".csv")
    except Exception as e:
        pass

    # 整合所有交易点
    if df_tpl_d is not None:
        df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60, df_tpl_d], ignore_index=True)
    else:
        df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60], ignore_index=True)

    # 将第一列转换为 datetime 格式
    df_tpl = df_tpl.set_index(df_tpl.columns[0])  # 使用第一列作为索引
    df_tpl.index.name = 'time'  # 将索引命名为 'time'
    # 修改时间列格式（索引）
    df_tpl.index = pd.to_datetime(df_tpl.index)
    # 修改时间列中含有00:00:00的部分为15:00:00
    df_tpl.index = df_tpl.index.map(
        lambda x: x.replace(hour=15, minute=0, second=0) if x.hour == 0 and x.minute == 0 and x.second == 0 else x)
    # 修改其余列的名称
    df_tpl.columns = ['price', 'signal']
    # 按索引（时间）排序
    df_tpl = df_tpl.sort_index()
    # 保存为新的 CSV 文件
    df_tpl.to_csv(tpl_filepath + assetList[0].assetsCode + "_concat" + ".csv")


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
    """
    backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_' +
                assetList[0].assetsCode + '_d.csv')
    signal_df_filepath = (RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_" +
                    assetList[0].assetsCode + "_concat" + ".csv")
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

                            if positive_count >= 2:
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
    signal_df.to_csv((RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_" +
                    assetList[0].assetsCode + "_concat_labeled" + ".csv"))

    print(assetList[0].assetsCode + "标注完成")


def pre_handle():
    """ """"""
    数据预处理
    A股、港股、美股、数字币。 每个市场风格不同，混合训练会降低特色，
        目前只用A股数据，沪深300+中证500=A股前800家上市公司
        涉及代码：HistoryData.py新增query_hs300_stocks、query_hs500_stocks，获取股票代码，
                get_ipo_date_for_stock、query_ipo_date 找出股票代码对应发行日期
                get_stock_from_code_csv 通过股票代码、发行日期，获取股票各级别历史行情
        待实验市场：港股、美股标普500、数字币市值前10
        数据来自证券宝，每个股票5种数据：日线、60m、30m、15m、5m。日线从该股发行日到2025年1月9日，分钟级最早为2019年1月2日。前复权，数据已压缩备份
    所有数据进行单级别回测，保留策略交易点，多进程运行
        目前策略：MACD+KDJ  （回归）
        涉及代码：旧代码在Run.py，5分钟bar转tick，给多级别同时用，回测一个股票5年要3小时。
                为提高效率，单级别运行，启动10线程，2台电脑，预计2、3天跑完4000个行情
        待实验策略：王立新ride-moon （趋势）  
                布林
                均线等，看是否比单纯指标有收益率提升
                （第三种方法、提前5天，直接抽特征自己发信号，不用判断当前信号是否有效）
    """
    allStockCode = pd.read_csv("./QuantData/a800_stocks.csv")
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['5', '15', '30', '60', 'd'],
                                             'stock',
                                             1)

        # 各级别交易点拼接在一起
        # concat_trade_point(assetList)
        # 过滤交易点
        # filter1(assetList)
        """
        把预处理数据转为 单变量定长或多变量定长
        组织数据
        依次遍历交易点，比如5分钟第一个交易点出现，此时拿到对应时间及label，按长度找到每个上级序列，加上label，还要沪深300
        """
        # 把有效交易点和原视数据结合，标注有效、无效
        # trans_point2label(asset)


def run_experiment():
    """模型训练"""
    # 1、构建弱学习器模型
    """
    建立model包，建个CNN.py
    先拿5分钟训练，数据进来，所有级别一起训练
    其他高级别训练放试验里
    """

    # 2、构建集成学习模型
    """实验"""
    # 1、策略不变，过滤方式变
    # 2、策略不变，过滤方式不变，超参变
    # 3、策略变等等


def run_live():
    pass


if __name__ == '__main__':
    pre_handle()  # 数据预处理
    # run_experiment()  # 实验回测
    # run_live()  # 实盘
