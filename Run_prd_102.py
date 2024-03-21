import requests
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime, time
from multiprocessing import Process
import sys

sys.path.append("/home/robotme")

import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
import RMQStrategy.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset
import RMQData.Bar_HistoryData as RMQBar_HistoryData
from RMQTool import Tools as RMTTools
from RMQTool import Message as RMTMessage


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

    req.close()

    # 收盘，保存买卖点信息，中午存一次，下午存一次
    for asset in assetList:
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_live") + "trade_point_list_" +
                          asset.inicatorEntity.IE_assetsCode + "_" +
                          asset.inicatorEntity.IE_timeLevel + ".csv", index=False, mode='a', header=False)


def start_process():
    processes = [Process(target=RMTMessage.FlashfeishuToken(),
                         args=())
                 ]

    for p in processes:
        # 启动进程
        p.start()

    for p in processes:
        p.join()
        p.close()


if __name__ == '__main__':
    while True:
        # 只能交易日的0~9:30之间，或交易日15~0之间，手动启
        workday_list = RMTTools.read_config("RMT", "workday_list") + "workday_list.csv"
        result = RMTTools.isWorkDay(workday_list, datetime.now().strftime("%Y-%m-%d"))  # 判断今天是不是交易日  元旦更新
        if result:  # 是交易日
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "早上开启进程")
            start_process()  # 今天运行，里面会等开盘时间到了才运行
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "中午进程停止，等下午")
            sleep(1800)  # 11:30休盘了，等半小时到12:30，开下午盘
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午开启进程")
            start_process()  # 开下午盘  第二次进入这个方法，所有进程都是新的，之前创建的，已经被close过了，不用担心内存溢出
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午进程停止，等明天")
            sleep(61200)  # 15点收盘，等17个小时，到第二天8点，重新判断是不是交易日
        else:  # 不是交易日
            sleep(86400)  # 直接等24小时，再重新判断
