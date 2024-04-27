import pandas as pd
from binance.spot import Spot
from datetime import datetime
from time import sleep
from multiprocessing import Process
import sys

sys.path.append("/home/RobotMeQ")
import RMQStrategy.Strategy as RMQStrategy
import RMQData.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset


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


    币安接口：https://github.com/binance/binance-connector-python  先去币安官网看看
    https://binance-docs.github.io/apidocs/spot/cn/#k

    etf 不同策略不同账户，马丁仓位
    cryp 不同策略，用不同子账户，币安可以建立不同子账户
        两个标的，一个b/u，一个e/b


    """
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    IEMultiLevel = RMQIndicator.IndicatorEntityMultiLevel()  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    proxies = {
        'http': 'http://127.0.0.1:33210',
        'https': 'http://127.0.0.1:33210',
    }
    client = Spot(proxies=proxies,timeout=3)
    #client = Spot(timeout=3)
    while True:
        try:
            # 每个级别都计算
            for asset in assetList:
                if asset.barEntity.timeLevel == "15":
                    k_interval = "15m"
                elif asset.barEntity.timeLevel == "60":
                    k_interval = "1h"
                elif asset.barEntity.timeLevel == "240":
                    k_interval = "4h"
                elif asset.barEntity.timeLevel == "d":
                    k_interval = "1d"

                # 币安的k线是实时的，省去了tick转bar，直接走策略
                df = pd.DataFrame(client.klines(symbol="BTCUSDT", interval=k_interval, limit=300),dtype=float)
                data = df.iloc[:, 0:6]  # 含头不含尾
                data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

                asset.barEntity.bar_DataFrame = data
                asset.indicatorEntity.tick_close = data.at[299, 'close']
                asset.indicatorEntity.tick_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(asset.indicatorEntity.tick_close)
                RMQStrategy.strategy(asset, strategy_result, IEMultiLevel)
        except Exception as e:
            print("Error happens", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e)
            sleep(3)  # 因为continue之后不走下面，所以再调一次
            continue
        sleep(100)


def start_process():
    processes = [Process(target=run_live_crypto,
                         args=(RMQAsset.asset_generator('BTCUSDT',
                                                        'BTC',
                                                        ['15', '60', '240', 'd'],
                                                        'crypto'),))
                 ]

    for p in processes:
        # 启动进程
        p.start()

    for p in processes:
        p.join()
        p.close()


if __name__ == '__main__':
    """
    实盘时用的国外服务器：
    1、dockerfile的COPY pip.conf、aliyun删除；
    2、requirements.txt加 binance-connector==2.0.0，删baostock==0.8.8
    """
    while True:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "开启进程")
        start_process()
