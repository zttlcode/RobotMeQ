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
from binance.spot import Spot
from RMQTool import Message as RMTMessage


# import sys
# 在cmd窗口python xxx.py 运行脚本时，自己写的from quant找不到quant，必须这样自定义一下python的系统变量，让python能找到
# sys.path.append(r"E:\\PycharmProjects\\robotme")
# 先运行此函数，再导自己的包
# from RobotMeQuant.RMQVisualized import DrawByMatplotlib as RMQMat


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
    ticks = RMQTick.get_ticks_for_backTesting(assetList[0].assetsCode, assetList[0].backtest_tick,
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
        backTest_result = asset.positionEntity.historyOrders
        print(asset.inicatorEntity.IE_assetsCode + "_" + asset.inicatorEntity.IE_timeLevel, backTest_result)
        # 计算每单收益
        RMQDrawPlot.draw_candle_orders(asset.backtest_bar, backTest_result, False)

        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backTest") + "trade_point_list_" +
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


def update_window_crypto(asset):
    asset.back_test_cut_row = asset.back_test_cut_row + 1
    asset.inicatorEntity.bar_DataFrame = asset.back_test_bar_data.iloc[asset.back_test_cut_row:asset.back_test_cut_row+asset.bar_num+1].copy()


def run_back_test_crypto(assetList):
    """
    实现思路：
        原来是最小级别tick，更新各个级别的bar
        现在是最小级别bar，更新各个级别的bar的close和时间窗口
        对于策略来说，我永远拿到当前级别的bar
    """
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    strategy_result.live = False
    IEMultiLevel = RMQIndicator.InicatorEntityMultiLevel()  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    # 初始化，读取回测数据
    for asset in assetList:
        # 读取每个级别的回测数据
        back_test_bar_data_tmp = pd.read_csv(asset.backtest_bar, parse_dates=['time'])
        # 遍历找到目标日期在哪一行，从这行往前截300条，这是初始数据，300窗口会一直往后移动
        # 所以，需要300数据+以后的所有数据，就是回测需要的全部数据
        cut_row = 0
        for index, row in back_test_bar_data_tmp.iterrows():  # 把读取的list数据遍历出来
            if row[0] == datetime(2020, 12, 27, 0, 0, 0) or (row[0] == datetime(2020, 12, 27) and asset.timeLevel == 'd'):
                break
            cut_row = cut_row + 1
        # 截好的300条 + 目标日期以后的所有数据，给bar_data，以后移动取bar_data的数据
        asset.back_test_bar_data = back_test_bar_data_tmp.iloc[cut_row - asset.bar_num:]
        # 滑动窗口从0开始往后滑动，窗口大小是bar_num
        asset.back_test_cut_row = 0

        # 然后把最初的300条，复制给bar_DataFrame，这里注意，除了最小级别15分钟，其他的要往后滑一位
        if asset.timeLevel == "15":
            # 15分钟是00，一进下面while，拿到00:15的数据
            # 其他级别如果也拿00，一进下面while，update_window不触发，那下面的for更新的是最后一个到00:00的bar，而不是00:00以后的第一个bar
            # 所以其他级别要在这里拿到 00:00后的第一个bar，然后在for里用15fen价格不断更新他，数量够了，窗口滑动，300bar被历史数据直接覆盖
            asset.inicatorEntity.bar_DataFrame = asset.back_test_bar_data.iloc[0:asset.bar_num+1].copy()  # 截取时能截50个，但含头不含尾
        else:
            asset.inicatorEntity.bar_DataFrame = asset.back_test_bar_data.iloc[1:asset.bar_num+2].copy()

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
            asset.inicatorEntity.tick_close = assetList[0].inicatorEntity.bar_DataFrame.tail(1).iloc[0, 4]
            asset.inicatorEntity.tick_time = assetList[0].inicatorEntity.bar_DataFrame.tail(1).iloc[0, 0]

            if asset.timeLevel != "15":
                # 非最小级别的，还要更新high、close
                asset.inicatorEntity.bar_DataFrame.tail(1).iloc[0, 2] = max(assetList[0].inicatorEntity.bar_DataFrame.tail(1).iloc[0, 2], asset.inicatorEntity.bar_DataFrame.tail(1).iloc[0, 2])
                asset.inicatorEntity.bar_DataFrame.tail(1).iloc[0, 3] = min(assetList[0].inicatorEntity.bar_DataFrame.tail(1).iloc[0, 3], asset.inicatorEntity.bar_DataFrame.tail(1).iloc[0, 3])

            RMQStrategy.strategy(asset.positionEntity, asset.inicatorEntity, asset.bar_num, strategy_result,
                                 IEMultiLevel)

        if assetList[0].inicatorEntity.bar_DataFrame.tail(1).iloc[0, 0] == datetime(2023, 6, 19, 0, 0, 0):
            break
        count = count + 1

    # 保存回测结果
    for asset in assetList:
        backTest_result = asset.positionEntity.historyOrders
        print(asset.inicatorEntity.IE_assetsCode + "_" + asset.inicatorEntity.IE_timeLevel, backTest_result)
        # 计算每单收益
        RMQDrawPlot.draw_candle_orders(asset.backtest_bar, backTest_result, False)

        # 保存买卖点信息
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backTest") + "trade_point_list_" +
                          asset.inicatorEntity.IE_assetsCode + "_" +
                          asset.inicatorEntity.IE_timeLevel + ".csv", index=False)


def run_live_crypto(assetList):
    """
    pip官方文档也教了pip安装官方源时，怎么用代理
    C:\\Users\Mr.EthanZ\AppData\Roaming\pip\pip.ini 文件里直接配置 proxy = http://127.0.0.1:33210 就可以了

[global]
proxy=http://127.0.0.1:33210

    之前pip配的国内镜像，pip.ini文件里是这样的
    [global]
    index-url = https://mirrors.aliyun.com/pypi/simple/

    [install]
    trusted-host = mirrors.aliyun.com
    以后都用代理，国内估计不再用了


    方法1：使用本地代理，
        代理费流量
        代理最连接数有问题
    方法2：申请国外服务器
       前提：需要国外银行卡，比如visa
       申请oracle cloud，免费的云服务器
       google cloud platform 谷歌的免费云服务器
       这两个都可以请求币安，同时能请求国内网址


    1：在A股基础上建立数字币系统
        币安接口：https://github.com/binance/binance-connector-python  先去币安官网看看
        https://binance-docs.github.io/apidocs/spot/cn/#k
    2：数字币能7*24盯盘，提醒我手动下单
    3：版本：数字币自动下单。

    etf 不同策略不同账户，马丁仓位
    cryp 不同策略，用不同子账户，币安可以建立不同子账户
        两个标的，一个b/u，一个e/b


    """
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    IEMultiLevel = RMQIndicator.InicatorEntityMultiLevel()  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    # proxies = {
    #     'http': 'http://127.0.0.1:33210',
    #     'https': 'http://127.0.0.1:33210',
    # }
    # client = Spot(proxies=proxies,timeout=3)
    client = Spot(timeout=3)
    while True:
        try:
            # 每个级别都计算
            for asset in assetList:
                if asset.timeLevel == "15":
                    k_interval = "15m"
                elif asset.timeLevel == "60":
                    k_interval = "1h"
                elif asset.timeLevel == "240":
                    k_interval = "4h"
                elif asset.timeLevel == "d":
                    k_interval = "1d"

                # 币安的k线是实时的，省去了tick转bar，直接走策略
                df = pd.DataFrame(client.klines(symbol="BTCUSDT", interval=k_interval, limit=300),dtype=float)
                data = df.iloc[:, 0:6]  # 含头不含尾
                data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

                asset.inicatorEntity.bar_DataFrame = data
                asset.inicatorEntity.tick_close = data.at[299, 'close']
                asset.inicatorEntity.tick_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                RMQStrategy.strategy(asset.positionEntity, asset.inicatorEntity, asset.bar_num, strategy_result,
                                     IEMultiLevel)
        except Exception as e:
            print("Error happens", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e)
            sleep(3)  # 因为continue之后不走下面，所以再调一次
            continue
        sleep(3)


def start_process():
    processes = [Process(target=RMTMessage.FlashfeishuToken(),args=()),
                  Process(target=run_live, args=(RMQAsset.asset_generator('510050', '上证50', ['5', '15', '30', '60', 'd'], 'ETF'),)),
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
    # run_live(RMQAsset.asset_generator('000001', '', ['5', '15', '30', '60', 'd'], 'index'))

    run_back_test_crypto(RMQAsset.asset_generator('BTCUSDT', 'BTC', ['15', '60', '240', 'd'], 'crypto'))

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
