import pandas as pd
from binance.spot import Spot
from datetime import datetime, time
from time import sleep
from RMQTool import Tools as RMTTools
import RMQVisualized.Draw_Matplotlib as RMQDrawPlot
import RMQStrategy.Strategy as RMQStrategy
import RMQStrategy.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset


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


if __name__ == '__main__':
    run_back_test_crypto(RMQAsset.asset_generator('BTCUSDT', 'BTC', ['15', '60', '240', 'd'], 'crypto'))
