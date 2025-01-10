import pandas as pd
from multiprocessing import Pool, current_process
import os
from datetime import datetime
import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
import RMQStrategy.Strategy_nature as RMQStrategy_nature
import RMQData.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset
import RMQVisualized.Draw_Matplotlib as RMQDrawPlot
from RMQTool import Tools as RMTTools

# import sys
# 在cmd窗口python xxx.py 运行脚本时，自己写的from quant找不到quant，必须这样自定义一下python的系统变量，让python能找到
# sys.path.append(r"E:\\PycharmProjects\\robotme")
# 先运行此函数，再导自己的包
# from RobotMeQ.RMQVisualized import DrawByMatplotlib as RMQMat


def run_back_test(assetList):
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    strategy_result.live = False
    IEMultiLevel = RMQIndicator.IndicatorEntityMultiLevel(assetList)  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    # 日线数据用5分钟的太慢，所以先加载回测开始日的，前250天日线数据，让日线指标更新上，方便其他级别使用日线指标
    # 2023 2 3 改进：除了5分钟级别，其他级别都先加载好250bar
    if len(assetList) > 1:
        # 如果是单级别，不走此函数；
        for asset in assetList:
            if asset.barEntity.timeLevel == '5':  # 5分钟的跳过，其他级别都要加载250个bar
                continue
            preTicks = []
            # 因为timeLevelList是从小到大放的，所以-1是最大级别
            preTicks = RMQTick.trans_bar_to_ticks(asset.assetsCode,
                                                  asset.barEntity.timeLevel,
                                                  asset.barEntity.backtest_bar,
                                                  preTicks)
            for preTick in preTicks:
                asset.barEntity.Tick = preTick
                asset.barEntity.bar_generator()  # 此时不用更新live的csv文件
                if asset.barEntity._init:  # 指标数据已生成，可以执行策略了
                    asset.update_indicatorDF_by_tick()

    # 1、回测bar数据转为tick
    # 因为timeLevelList是从小到大放的，所以0是最小级别
    ticks = RMQTick.get_ticks_for_backtesting(assetList[0].assetsCode,
                                              assetList[0].barEntity.backtest_tick,
                                              assetList[0].barEntity.backtest_bar,
                                              assetList[0].barEntity.timeLevel)
    # 2、回测数据在此函数内疯狂循环
    for tick in ticks:
        # 每个级别都用tick
        for asset in assetList:
            asset.barEntity.Tick = tick
            asset.barEntity.bar_generator()  # 创建并维护bar，生成指标数据
            if asset.barEntity._init:  # 指标数据已生成，可以执行策略了
                asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看
                RMQStrategy.strategy(asset,
                                     strategy_result,
                                     IEMultiLevel)  # 整个系统最耗时的在这里，15毫秒

    # 返回结果
    for asset in assetList:
        backtest_result = asset.positionEntity.historyOrders
        if 0 != len(asset.positionEntity.historyOrders):
            print(asset.indicatorEntity.IE_assetsCode + "_" + asset.indicatorEntity.IE_timeLevel, backtest_result)
            # 计算每单收益
            RMQDrawPlot.draw_candle_orders(asset.barEntity.backtest_bar, backtest_result, False)
        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backtest")
                          + "trade_point_list_"
                          + asset.indicatorEntity.IE_assetsCode
                          + "_"
                          + asset.indicatorEntity.IE_timeLevel
                          + ".csv", index=False)


def run_back_test_no_tick(assetList):
    """
    注意：我之前代码的回测是多级别，从5分钟级别转tick，各级别再跟着tick转bar，这是因为策略要判断当时日线指标，但现在不用
    加日线过滤出了子集，不加就是全集，交易点位变多了，影响不大。
    现在为了加速，不用tick，就省去了tick转bar。要做的修改如下
        原来tick循环时，每个bar要更新 indicatorEntity 和 barEntity.bar_DataFrame
        然后在策略中，用barEntity算指标
    """
    for asset in assetList:
        start = datetime.now()
        # 读取这个资产、这个级别的历史数据
        backtest_bar_data = pd.read_csv(asset.barEntity.backtest_bar, parse_dates=['time'])
        # 遍历他
        window_size = 250
        for start in range(0, len(backtest_bar_data) - window_size + 1):  # 滑动窗口，从每一行开始
            end = start + window_size
            # 把bar给barEntity
            asset.barEntity.bar_DataFrame = backtest_bar_data.iloc[start:end]  # 提取窗口内的数据
            # 把最新一行的各种值，给indicatorEntity
            # 每次tick都在策略中更新指标
            asset.indicatorEntity.tick_high = asset.barEntity.bar_DataFrame.at[end-1, 'high']
            asset.indicatorEntity.tick_low = asset.barEntity.bar_DataFrame.at[end-1, 'low']
            asset.indicatorEntity.tick_close = asset.barEntity.bar_DataFrame.at[end-1, 'close']
            asset.indicatorEntity.tick_time = asset.barEntity.bar_DataFrame.at[end-1, 'time']
            asset.indicatorEntity.tick_volume = asset.barEntity.bar_DataFrame.at[end-1, 'volume']
            # 把一个标的 一个级别的所有数据回测，记录交易点
            # 这个策略做了改动，strategy中的判断日线被删除了
            RMQStrategy_nature.strategy(asset, None, None)  # 整个系统最耗时的在这里，15毫秒
        # backtest_result = asset.positionEntity.historyOrders
        # if 0 != len(asset.positionEntity.historyOrders):
        #     print(asset.indicatorEntity.IE_assetsCode + "_" + asset.indicatorEntity.IE_timeLevel, backtest_result)
        #     # 计算每单收益
        #     RMQDrawPlot.draw_candle_orders(asset.barEntity.backtest_bar, backtest_result, False)
        print(asset.indicatorEntity.IE_assetsCode + "_" + asset.indicatorEntity.IE_timeLevel + "结束")
        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backtest")
                          + "trade_point_list_"
                          + asset.indicatorEntity.IE_assetsCode
                          + "_"
                          + asset.indicatorEntity.IE_timeLevel
                          + ".csv", index=False)


def chunk_dataframe(df, num_chunks):
    """
    将 DataFrame 分成指定数量的块。
    """
    chunk_size = len(df) // num_chunks
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def run_backTest_multip(data_chunk):
    """
    处理数据块的函数。
    每个进程会调用这个函数来处理指定的数据块。
    """
    process_name = current_process().name  # 获取当前进程的名称
    process_id = os.getpid()  # 获取当前进程 ID
    for _, row in data_chunk.iterrows():
        print(f"Process {process_name} (PID {process_id}) is processing: {row['code'][3:]}")
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['5', '15', '30', '60', 'd'],
                                             'stock',
                                             1)
        run_back_test_no_tick(assetList)  # 0:02:29.502122 新回测，不转tick


if __name__ == '__main__':
    """
    要想运行多级别，列表里加一个时间级别就行
    要想运行单级别，列表里就一个时间级别就行
    每个资产都有5个级别，5、15、30、60、日线  新增级别需要修改 decide_time_level()、trans_bar_to_ticks()
    
    注意：实盘不能盘中启动！宕机只能手段补数据
    快开盘时启动。这样[0]保存的是昨天最后一个bar的数据，开盘插入新bar，更新新bar数据，第二个bar开启时，保存这个新bar
    如果盘中重启，[0]还是上一个bar的数据，因为不满足整点，[0]的数据就会被更新，无法记录当前bar的最高最低值，和成交量
    如果盘中启动，收盘后，需要改第一个bar的数据，和其余三个bar的成交量。幸好我是小时线
    """
    # run_back_test(RMQAsset.asset_generator('000001',
    #                                        '上证',
    #                                        ['5', '15', '30', '60', 'd'],
    #                                        'index',
    #                                        1))
    allStockCode = pd.read_csv("./QuantData/a800_stocks.csv")
    # 多个并行
    # 确定进程数量和数据块
    num_processes = 20
    data_chunks = chunk_dataframe(allStockCode, num_processes)  # 把300个股票分给20个进程并行处理
    # 使用 multiprocessing 开启进程池
    with Pool(num_processes) as pool:
        pool.map(run_backTest_multip, data_chunks)


    # 实验一个
    # for index, row in allStockCode.iterrows():
    #     assetList = RMQAsset.asset_generator(row['code'][3:],
    #                                          row['code_name'],
    #                                          ['5', '15', '30', '60', 'd'],
    #                                          'stock',
    #                                          1)
    #
    #     # run_back_test(assetList)  # 0:18:27.437876 旧回测，转tick，运行时长  加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
    #     run_back_test_no_tick(assetList)  # 0:02:29.502122 新回测，不转tick
    #     break


