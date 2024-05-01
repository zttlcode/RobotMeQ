import pandas as pd
import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
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
    # run_back_test(RMQAsset.asset_generator('601012', '', ['5', '15', '30', '60', 'd'], 'stock'))
    run_back_test(RMQAsset.asset_generator('000001',
                                           '上证',
                                           ['5', '15', '30', '60', 'd'],
                                           'index',
                                           1))
