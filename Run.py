import requests
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime, time
import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
import RMQStrategy.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset
import RMQVisualized.Draw_Matplotlib as RMQDrawPlot
import RMQData.Bar_HistoryData as RMQBar_HistoryData
from RMQTool import Tools as RMTTools
from multiprocessing import Process

from RMQTool import Message as RMTMessage


# import sys
# 在cmd窗口python xxx.py 运行脚本时，自己写的from quant找不到quant，必须这样自定义一下python的系统变量，让python能找到
# sys.path.append(r"E:\\PycharmProjects\\robotme")
# 先运行此函数，再导自己的包
# from RobotMeQ.RMQVisualized import DrawByMatplotlib as RMQMat


def run_back_test(assetList):
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    strategy_result.live = False
    IEMultiLevel = RMQIndicator.InicatorEntityMultiLevel()  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    # 日线数据用5分钟的太慢，所以先加载回测开始日的，前250天日线数据，让日线指标更新上，方便其他级别使用日线指标
    # 2023 2 3 改进：除了5分钟级别，其他级别都先加载好250bar
    if len(assetList) > 1:
        # 如果是单级别，不走此函数；
        for asset in assetList:
            if asset.timeLevel == '5':  # 5分钟的跳过，其他级别都要加载250个bar
                continue
            preTicks = []
            # 因为timeLevelList是从小到大放的，所以-1是最大级别
            preTicks = RMQTick.trans_bar_to_ticks(asset.assetsCode, asset.timeLevel, asset.backtest_bar, preTicks)
            for preTick in preTicks:
                asset.Tick = preTick
                asset.bar_generator()  # 此时不用更新live的csv文件
                if asset._init:  # 指标数据已生成，可以执行策略了
                    asset.update_indicatorDF_by_tick()

    # 1、回测bar数据转为tick
    # 因为timeLevelList是从小到大放的，所以0是最小级别
    ticks = RMQTick.get_ticks_for_backtesting(assetList[0].assetsCode, assetList[0].backtest_tick,
                                              assetList[0].backtest_bar, assetList[0].timeLevel)
    # 2、回测数据在此函数内疯狂循环
    for tick in ticks:
        # 每个级别都用tick
        for asset in assetList:
            asset.Tick = tick
            asset.bar_generator()  # 创建并维护bar，生成指标数据
            if asset._init:  # 指标数据已生成，可以执行策略了
                asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看
                RMQStrategy.strategy(asset.positionEntity, asset.inicatorEntity, asset.bar_num, strategy_result,
                                     IEMultiLevel)  # 整个系统最耗时的在这里，15毫秒

    # 返回结果
    for asset in assetList:
        backtest_result = asset.positionEntity.historyOrders
        print(asset.inicatorEntity.IE_assetsCode + "_" + asset.inicatorEntity.IE_timeLevel, backtest_result)
        # 计算每单收益
        RMQDrawPlot.draw_candle_orders(asset.backtest_bar, backtest_result, False)

        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_" +
                          asset.inicatorEntity.IE_assetsCode + "_" +
                          asset.inicatorEntity.IE_timeLevel + ".csv", index=False)


def run_live(assetList):
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    IEMultiLevel = RMQIndicator.InicatorEntityMultiLevel()  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    for asset in assetList:
        # 1、加载实盘历史live_bar数据转为tick
        ticks = []
        # 因为timeLevelList是从小到大放的，所以0是最小级别
        ticks = RMQTick.trans_bar_to_ticks(asset.assetsCode, asset.timeLevel, asset.live_bar, ticks)
        for tick in ticks:
            asset.Tick = tick
            asset.bar_generator()  # 此时不用更新live的csv文件
            if asset._init:  # 指标数据已生成，可以执行策略了
                asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看

    # 2、准备工作完成，在这里等开盘
    # 闭市期间，程序关闭，所以下午是个新bar.(不关闭的话，中午的一小时里数据没用，但bar已生成，还得再清理，更麻烦)
    while datetime.now().time() < time(9, 30) or time(11, 31) < datetime.now().time() < time(13) or time(15, 1) \
            <= datetime.now().time():
        sleep(1)

    # 3、实盘开启，此参数只控制bar生成的部分操作
    for asset in assetList:
        asset.isLiveRunning = True

    # 获取request连接池，用连接池去请求，省资源
    req = requests.sessions.Session()

    while time(9, 30) < datetime.now().time() < time(11, 34) or time(13) < datetime.now().time() < time(15, 4):
        # 11:29:57程序直接停了，估计是判断11:30:00直接结束，但我需要它进到11：30，才能保存最后一个bar，所以改成31分
        try:
            # 我本地不会出错，只有这个地方可能报请求超时，所以加个try
            resTick = RMQTick.getTick(req, assetList[0].assetsCode, assetList[0].assetsType)  # 获取实时股价
        except Exception as e:
            print("Error happens", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e)
            sleep(3)  # 因为continue之后不走下面，所以再调一次
            continue

        for asset in assetList:
            asset.Tick = resTick
            asset.bar_generator()  # 更新live的文件

        if time(11, 30) < resTick[0].time() < time(11, 34) or time(15) <= resTick[0].time() < time(15, 4):
            # 到收盘时间，最后一个bar已写入csv，此时new了新bar，已经没用了，就不影响后续，只等程序结束自动销毁
            print("收盘时间到，程序停止", datetime.now().time(), resTick[0].time())
            # 每天下午收盘后，整理当日bar数据
            if time(15) <= resTick[0].time():
                # 1、更新日线bar数据
                resTickForDay = RMQTick.getTickForDay(req, assetList[-1].assetsCode, assetList[-1].assetsType)
                data_list = [resTickForDay[0].strftime('%Y-%m-%d'), resTickForDay[1], resTickForDay[2],
                             resTickForDay[3], resTickForDay[4], resTickForDay[5]]
                # print("日线bar已更新：", data_list)
                # 输入的list为长度6的list（6行rows），而DataFrame需要的是6列(columns)的list。
                # 因此，需要将test_list改为（1*6）的list就可以了。
                data_list = np.array(data_list).reshape(1, 6)
                result = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                result.loc[:, 'time'] = pd.to_datetime(result.loc[:, 'time'])
                # 输出到csv文件
                result.to_csv(assetList[-1].live_bar, index=False, mode='a', header=False)

                # 2、把实盘数据截为250，这样大小永远固定
                for asset in assetList:
                    bar_data = pd.read_csv(asset.live_bar)
                    windowDF = RMQBar_HistoryData.cut_by_bar_num(bar_data, asset.bar_num)
                    windowDF.to_csv(asset.live_bar, index=0)
            break
        else:
            for asset in assetList:
                if asset._init:  # 指标数据已生成，可以执行策略了
                    asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看
                    RMQStrategy.strategy(asset.positionEntity, asset.inicatorEntity, asset.bar_num, strategy_result,
                                         IEMultiLevel)
        sleep(3)  # 3秒调一次

    # 2023 04 start_process中的join卡住了，调用链显示一个线程在等网络io，唯一的可能就是超时后req线程没有释放，所以这里关闭线程池试试
    # 验证后确实是这个原因
    req.close()

    # 收盘，保存买卖点信息，中午存一次，下午存一次
    for asset in assetList:
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_live") + "trade_point_list_" +
                          asset.inicatorEntity.IE_assetsCode + "_" +
                          asset.inicatorEntity.IE_timeLevel + ".csv", index=False, mode='a', header=False)


def start_process():
    """
    只需要记住一点，要想实现多线程， target=方法名/函数名，后不能带括号（）。
1、不带括号时，调用的是这个函数本身 ，是整个函数体，是一个函数对象，不需等该函数执行完成；
2、带括号（此时必须传入需要的参数），调用的是函数的return结果，需要等待函数执行完成的结果。
    """
    processes = [ Process(target=run_live, args=(RMQAsset.asset_generator('510050', '上证50', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159915', '创业板', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('510300', '沪深300指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('510500', '中证500指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512100', '中证1000指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('588000', '科创50', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159920', '恒生', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159941', '纳指', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512690', '酒', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512480', '半导体', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('515030', '新能源车', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('513050', '中概互联', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('513060', '恒生医疗', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('515790', '光伏', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('516970', '基建', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512660', '军工', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159611', '电力', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512200', '地产', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512170', '医疗', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512800', '银行', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512980', '传媒', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('512880', '证券', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('515220', '煤炭', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159766', '旅游', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('159865', '养殖', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                  Process(target=run_live, args=(RMQAsset.asset_generator('518880', '黄金', ['5', '15', '30', '60', 'd'], 'ETF'),))]

    for p in processes:
        # 启动进程
        p.start()

    for p in processes:
        p.join()
        """
        
        
        等待工作进程结束，只要有一个进程没运行完，所有进程都会卡join等
        第一个进程join，它执行完后，join结束，执行close，回收进程资源，然后第二个join,
        由于我每个进程结束时间差不多，所以第二个刚一join就马上close了，直到所有都完成
        如果第二三个快，第一个慢，那么第二三个执行完，也要在这里等第一个，只有第一个join完了，for循环才去join第二三
        
        所以，多个进程并行，没有问题，先后运行结束，也没问题，只是大家要在这里等第一个进程，或者说等for循环遇见的慢的那个
        由于我的进程开始结束时间最多差一二分钟，所以没啥影响
        直到processes里全部进程都运行完，再出循环，跳出此函数，继续执行主函数
        
        如果某个进程卡住，win里可以在 资源管理器，右键进程查看详细信息，然后右键分析调用链，能看到进程卡住原因，手动结束，可以继续运行
        但要注意分辨它不是主进程
        """
        p.close()


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
    # 回测与实盘，一个实例只会运行其中一个
    # assetsType ： stock index ETF 甚至 crypto

    # run_back_test(RMQAsset.asset_generator('601012', '', ['5', '15', '30', '60', 'd'], 'stock'))
    # run_back_test(RMQAsset.asset_generator('000001', '', ['5', '15', '30', '60', 'd'], 'index'))

    # run_live(RMQAsset.asset_generator('399006', '', ['5', 'd'], 'index'))
    run_live(RMQAsset.asset_generator('000001', '', ['5', '15', '30', '60', 'd'], 'index'))


    """
    while True:
        # 只能交易日的0~9:30之间，或交易日15~0之间，手动启
        workday_list = RMTTools.read_config("RMT", "workday_list") + "workday_list.csv"
        result = RMTTools.isWorkDay(workday_list, datetime.now().strftime("%Y-%m-%d"))  # 判断今天是不是交易日  元旦更新
        if result:  # 是交易日
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "早上开启进程")
            start_process()  # 今天运行，里面会等开盘时间到了才运行
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "中午进程停止，等下午")
            sleep(1800)  # 11:30休盘了，等1小时到12:30，开下午盘
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午开启进程")
            start_process()  # 开下午盘  第二次进入这个方法，所有进程都是新的，之前创建的，已经被close过了，不用担心内存溢出
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午进程停止，等明天")
            sleep(61200)  # 15点收盘，等17个小时，到第二天8点，重新判断是不是交易日
        else:  # 不是交易日
            sleep(86400)  # 直接等24小时，再重新判断
    """
