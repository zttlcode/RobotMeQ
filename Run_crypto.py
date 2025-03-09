import pandas as pd
from datetime import datetime
from RMQTool import Tools as RMTTools
import RMQStrategy.Strategy as RMQStrategy
import RMQData.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset


def update_window_crypto(asset):
    asset.back_test_cut_row = asset.back_test_cut_row + 1
    asset.barEntity.bar_DataFrame = asset.back_test_bar_data.iloc[asset.back_test_cut_row:asset.back_test_cut_row + asset.barEntity.bar_num + 1].copy()


def run_back_test_crypto(assetList, strategy_name):
    """
    实现思路：
        原来是最小级别tick，更新各个级别的bar
        现在是最小级别bar，更新各个级别的bar的close和时间窗口
        对于策略来说，我永远拿到当前级别的bar
    """
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    strategy_result.live = False
    IEMultiLevel = RMQIndicator.IndicatorEntityMultiLevel(assetList)  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    # 初始化，读取回测数据
    for asset in assetList:
        # 读取每个级别的回测数据
        back_test_bar_data_tmp = pd.read_csv(asset.barEntity.backtest_bar, parse_dates=['time'])
        # 遍历找到目标日期在哪一行，从这行往前截300条，这是初始数据，300窗口会一直往后移动
        # 所以，需要300数据+以后的所有数据，就是回测需要的全部数据
        cut_row = 0
        for index, row in back_test_bar_data_tmp.iterrows():  # 把读取的list数据遍历出来
            if (row[0] == datetime(2020, 12, 27, 0, 0, 0)
                    or (row[0] == datetime(2020, 12, 27)
                        and asset.barEntity.timeLevel == 'd')):
                break
            cut_row = cut_row + 1
        # 截好的300条 + 目标日期以后的所有数据，给bar_data，以后移动取bar_data的数据
        asset.back_test_bar_data = back_test_bar_data_tmp.iloc[cut_row - asset.barEntity.bar_num:]
        # 滑动窗口从0开始往后滑动，窗口大小是bar_num
        asset.back_test_cut_row = 0

        # 然后把最初的300条，复制给bar_DataFrame，这里注意，除了最小级别15分钟，其他的要往后滑一位
        if asset.barEntity.timeLevel == "15":
            # 15分钟是00，一进下面while，拿到00:15的数据
            # 其他级别如果也拿00，一进下面while，update_window不触发，那下面的for更新的是最后一个到00:00的bar，而不是00:00以后的第一个bar
            # 所以其他级别要在这里拿到 00:00后的第一个bar，然后在for里用15fen价格不断更新他，数量够了，窗口滑动，300bar被历史数据直接覆盖
            asset.barEntity.bar_DataFrame = asset.back_test_bar_data.iloc[0:asset.barEntity.bar_num + 1].copy()  # 截取时能截50个，但含头不含尾
        else:
            asset.barEntity.bar_DataFrame = asset.back_test_bar_data.iloc[1:asset.barEntity.bar_num + 2].copy()

    # 初始化完成，现在每个级别的asset都有自己的回测数据bar_data，和窗口数据bar_DataFrame
    # 接下来按最小级别15分钟，开始回测
    count = 1  # 用于控制更新非最小级别bar
    while True:
        # 每1次，15分钟读新300
        update_window_crypto(assetList[0])
        if count % 4 == 0:
            # 每4次，60分钟读新300
            update_window_crypto(assetList[1])
        if count % 16 == 0:  # 不能用else，不然后面都走不了
            # 每16次，240分钟读新300
            update_window_crypto(assetList[2])
        if count % 96 == 0:
            # 每96次，d读新300
            update_window_crypto(assetList[3])
            count = 0
        # 每个级别都计算
        for asset in assetList:
            # 每次都把最小级别的close更新给所有级别
            asset.indicatorEntity.tick_close = assetList[0].barEntity.bar_DataFrame.tail(1).iloc[0, 4]
            asset.indicatorEntity.tick_time = assetList[0].barEntity.bar_DataFrame.tail(1).iloc[0, 0]

            if asset.barEntity.timeLevel != "15":
                # 非最小级别的，还要更新high、close
                asset.barEntity.bar_DataFrame.tail(1).iloc[0, 2] = max(assetList[0].barEntity.bar_DataFrame.tail(1).iloc[0, 2], asset.barEntity.bar_DataFrame.tail(1).iloc[0, 2])
                asset.barEntity.bar_DataFrame.tail(1).iloc[0, 3] = min(assetList[0].barEntity.bar_DataFrame.tail(1).iloc[0, 3], asset.barEntity.bar_DataFrame.tail(1).iloc[0, 3])

            RMQStrategy.strategy(asset, strategy_result, IEMultiLevel, strategy_name)

        if assetList[0].barEntity.bar_DataFrame.tail(1).iloc[0, 0] == datetime(2023, 6, 19, 0, 0, 0):
            break
        count = count + 1

    # 保存回测结果
    for asset in assetList:
        backtest_result = asset.positionEntity.historyOrders
        print(asset.indicatorEntity.IE_assetsCode + "_" + asset.indicatorEntity.IE_timeLevel, backtest_result)
        # 计算每单收益
        if len(backtest_result) != 0:
            # 计算每单收益
            orders_df = pd.DataFrame(backtest_result).T  # DataFrame之后是矩阵样式，列标题是字段名，行标题是每个订单，加T是转置，列成了每单，跟excel就一样了
            print(orders_df.loc[:, 'pnl'].sum())  # 显示总收益
        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backtest_fuzzy")
                          + asset.assetsMarket
                          + "_"
                          + asset.indicatorEntity.IE_assetsCode
                          + "_"
                          + asset.indicatorEntity.IE_timeLevel + ".csv", index=False)


if __name__ == '__main__':
    """
    实盘时用的国外服务器：
    1、dockerfile的COPY pip.conf、aliyun删除；
    2、requirements.txt加 binance-connector==2.0.0，删baostock==0.8.8
    """
    run_back_test_crypto(RMQAsset.asset_generator('BTCUSDT', 'BTC',
                                                  ['15', '60', '240', 'd'], 'crypto',
                                                  0, 'crypto'), 'tea_radical')
