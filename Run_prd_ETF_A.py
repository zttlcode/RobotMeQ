import requests
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime, time
from multiprocessing import Process
import sys
import os

sys.path.append("/home/RobotMeQ")

import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
import RMQData.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset
import RMQData.HistoryData as RMQBar_HistoryData
from RMQTool import Tools as RMTTools


def run_live(assetList, strategy_name):
    strategy_result = RMQStrategy.StrategyResultEntity()  # 收集多级别行情信息，推送消息
    IEMultiLevel = RMQIndicator.IndicatorEntityMultiLevel(assetList)  # 多级别的指标要互相交流，所以通过这个公共指标对象交流

    for asset in assetList:
        # 1、加载实盘历史live_bar数据转为tick
        ticks = []
        # 因为timeLevelList是从小到大放的，所以0是最小级别
        ticks = RMQTick.trans_bar_to_ticks(asset.assetsCode,
                                           asset.barEntity.timeLevel,
                                           asset.barEntity.live_bar,
                                           ticks)
        for tick in ticks:
            asset.barEntity.Tick = tick
            asset.barEntity.bar_generator()  # 此时不用更新live的csv文件
            if asset.barEntity._init:  # 指标数据已生成，可以执行策略了
                asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看

    # 2、准备工作完成，在这里等开盘
    # 闭市期间，程序关闭，所以下午是个新bar.(不关闭的话，中午的一小时里数据没用，但bar已生成，还得再清理，更麻烦)
    while (datetime.now().time() < time(9, 30)
           or time(11, 31) < datetime.now().time() < time(13)
           or time(15, 1) <= datetime.now().time()):
        sleep(1)

    # 3、实盘开启，此参数只控制bar生成的部分操作
    for asset in assetList:
        asset.barEntity.isLiveRunning = True

    # 获取request连接池，用连接池去请求，省资源
    req = requests.sessions.Session()

    while (time(9, 30) < datetime.now().time() < time(11, 34)
           or time(13) < datetime.now().time() < time(15, 4)):
        # 11:29:57程序直接停了，估计是判断11:30:00直接结束，但我需要它进到11：30，才能保存最后一个bar，所以改成31分
        try:
            # 我本地不会出错，只有这个地方可能报请求超时，所以加个try
            resTick = RMQTick.getTick(req, assetList[0].assetsCode, assetList[0].assetsType)  # 获取实时股价
        except Exception as e:
            print("Error happens", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e)
            sleep(1)  # 因为continue之后不走下面，所以再调一次
            continue

        for asset in assetList:
            asset.barEntity.Tick = resTick
            asset.barEntity.bar_generator()  # 更新live的文件

        if (time(11, 30) < resTick[0].time() < time(11, 34)
                or time(15) <= resTick[0].time() < time(15, 4)):
            # 到收盘时间，最后一个bar已写入csv，此时new了新bar，已经没用了，就不影响后续，只等程序结束自动销毁
            print("收盘时间到，程序停止", datetime.now().time(), resTick[0].time())
            # 每天下午收盘后，整理当日bar数据
            if time(15) <= resTick[0].time():
                # 1、更新日线bar数据
                resTickForDay = RMQTick.getTickForDay(req, assetList[-1].assetsCode, assetList[-1].assetsType)
                data_list = [resTickForDay[0].strftime('%Y-%m-%d'),
                             resTickForDay[1],
                             resTickForDay[2],
                             resTickForDay[3],
                             resTickForDay[4],
                             resTickForDay[5]]
                # print("日线bar已更新：", data_list)
                # 输入的list为长度6的list（6行rows），而DataFrame需要的是6列(columns)的list。
                # 因此，需要将test_list改为（1*6）的list就可以了。
                data_list = np.array(data_list).reshape(1, 6)
                result = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                result.loc[:, 'time'] = pd.to_datetime(result.loc[:, 'time'])
                # 输出到csv文件
                result.to_csv(assetList[-1].barEntity.live_bar, index=False, mode='a', header=False)

                # 2、把实盘数据截为250，这样大小永远固定
                for asset in assetList:
                    bar_data = pd.read_csv(asset.barEntity.live_bar)
                    windowDF = RMQBar_HistoryData.cut_by_bar_num(bar_data, asset.barEntity.bar_num)
                    windowDF.to_csv(asset.barEntity.live_bar, index=0)
            break
        else:
            for asset in assetList:
                if asset.barEntity._init:  # 指标数据已生成，可以执行策略了
                    asset.update_indicatorDF_by_tick()  # 必须在此更新，不然就要把5个值作为参数传递，不好看
                    RMQStrategy.strategy(asset, strategy_result, IEMultiLevel, strategy_name)
        sleep(1)  # 1秒调一次

    # 2023 04 start_process中的join卡住了，调用链显示一个线程在等网络io，唯一的可能就是超时后req线程没有释放，所以这里关闭线程池试试
    # 验证后确实是这个原因
    req.close()

    # 收盘，保存买卖点信息，中午存一次，下午存一次
    for asset in assetList:
        if asset.positionEntity.trade_point_list:  # 不为空，则保存
            df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
            item = 'trade_point_live_' + strategy_name
            directory = RMTTools.read_config("RMQData", item)
            os.makedirs(directory, exist_ok=True)
            df_tpl.to_csv(directory
                          + asset.assetsMarket
                          + "_"
                          + asset.indicatorEntity.IE_assetsCode
                          + "_"
                          + asset.indicatorEntity.IE_timeLevel
                          + ".csv", index=False, mode='a', header=False)


def start_process():
    strategy_name = 'tea_radical'
    timeLevelList = ['5', '15', '30', '60', 'd']
    """
    只需要记住一点，要想实现多线程， target=方法名/函数名，后不能带括号（）。
    1、不带括号时，调用的是这个函数本身 ，是整个函数体，是一个函数对象，不需等该函数执行完成；
    2、带括号（此时必须传入需要的参数），调用的是函数的return结果，需要等待函数执行完成的结果。
    """
    processes = [Process(target=run_live,
                         args=(RMQAsset.asset_generator('510050',
                                                        '上证50',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159915',
                                                        '创业板',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('510300',
                                                        '沪深300指数',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('563300',
                                                        '中证2000指数',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('588000',
                                                        '科创50',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512690',
                                                        '酒',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515030',
                                                        '新能源车',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515790',
                                                        '光伏',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('516970',
                                                        '基建',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512660',
                                                        '军工',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159611',
                                                        '电力',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512170',
                                                        '医疗',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512800',
                                                        '银行',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515220',
                                                        '煤炭',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159766',
                                                        '旅游',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159865',
                                                        '养殖',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159996',
                                                        '家电',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512480',
                                                        '半导体',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159819',
                                                        '人工智能',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('562500',
                                                        '机器人',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159869',
                                                        '游戏',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515880',
                                                        '通信',
                                                        timeLevelList,
                                                        'ETF',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159985',
                                                        '豆粕',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159980',
                                                        '有色',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159981',
                                                        '能源化工',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513360',
                                                        '教育',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('518880',
                                                        '黄金',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159920',
                                                        '恒生',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513130',
                                                        '恒生科技',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513060',
                                                        '恒生医疗',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159941',
                                                        '纳指',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159509',
                                                        '纳指科技',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513030',
                                                        '德国',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159866',
                                                        '日经',
                                                        timeLevelList,
                                                        'ETF',
                                                        0, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('601658',
                                                        '邮储银行',
                                                        timeLevelList,
                                                        'stock',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('600905',
                                                        '三峡能源',
                                                        timeLevelList,
                                                        'stock',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('601598',
                                                        '中国外运',
                                                        timeLevelList,
                                                        'stock',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('601868',
                                                        '中国能建',
                                                        timeLevelList,
                                                        'stock',
                                                        1, 'A'), strategy_name,)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('600690',
                                                        '海尔智家',
                                                        timeLevelList,
                                                        'stock',
                                                        1, 'A'), strategy_name,))
                 ]

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
--部署代码
    一个新服务器，安装docker : yum update更新一下库
    然后执行 curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    执行 service docker start 启动服务      systemctl enable docker 开机启动
    把RobotMeQ放到服务器~目录下，dockerfile拿出来放~目录，运行 docker build -t python:3.8.2 .
    我的笔记本build没问题，但台式机build的半个小时一直没好，我换了docker的镜像源才行（菜鸟教程有教），
        下面是我的阿里云镜像原，其他国内镜像都访问不了了
        在 /etc/docker/daemon.json 中写入如下内容（如果文件不存在请新建该文件）
{
"registry-mirrors":["https://0hc2yp52.mirror.aliyuncs.com"]
}
    # 如果dockerfile最后一行地pip.conf没用，就直接用下面这个运行
        pip install -r requirements.txt --trusted-host mirrors.aliyun.com
        # 就算pip 单独安装包，也要加--trusted-host mirrors.aliyun.com
    之后重新启动服务：
    $ sudo systemctl daemon-reload
    $ sudo systemctl restart docker
    再执行上面的docker build就可以了
    build完，docker images就能看到镜像，然后启动镜像为容器
    docker run -it 2023c3642f33 /bin/bash
    以后就可以通过exec进入了
    
    我的docker容器默认是Debian，要想装less和vim，需要运行
    apt update
    apt install -y less vim
    
    阿里云 2024-4月~2026-4月  zhaot1993@qq.com
    docker start 26dbf8178821
    docker exec -it 26dbf8178821 /bin/bash
        nohup python -u /home/RobotMeQ/Run_prd_ETF_A.py >> /home/log.out 2>&1 &
    tail -f /home/log.out
    docker cp /root/RobotMeQ 26dbf8178821:/home/RobotMeQ
    docker cp /root/RobotMeQ/Run_prd_ETF_A.py 26dbf8178821:/home/RobotMeQ/Run_prd_ETF_A.py
    docker cp /root/RobotMeQ/requirements.txt 26dbf8178821:/home/RobotMeQ/requirements.txt
    docker cp /root/RobotMeQ/QuantData/live 26dbf8178821:/home/RobotMeQ/QuantData/live2
    docker cp 26dbf8178821:/home/RobotMeQ/QuantData/live /root/RobotMeQ/QuantData/live
    docker cp 26dbf8178821:/home/RobotMeQ/QuantData/trade_point_live /root/RobotMeQ/QuantData/trade_point_live
    docker cp 26dbf8178821:/home/RobotMeQ/QuantData/position_historyOrders /root/RobotMeQ/QuantData/position_historyOrders
    docker cp 26dbf8178821:/home/RobotMeQ/QuantData/position_currentOrders /root/RobotMeQ/QuantData/position_currentOrders
    docker cp 26dbf8178821:/home/log.out /root/RobotMeQ/QuantData/log.out

    腾讯云  287151402@qq.com
    docker start 5c239d668666
    docker exec -it 5c239d668666 /bin/bash
    docker cp /root/RobotMeQ 5c239d668666:/home/RobotMeQ
    docker cp /root/RobotMeQ/RMQStrategy/Strategy.py 5c239d668666:/home/RobotMeQ/RMQStrategy/Strategy.py
    docker cp /root/RobotMeQ/QuantData/live 5c239d668666:/home/RobotMeQ/QuantData/live
    docker cp /root/RobotMeQ/Configs/config_prd.ini 5c239d668666:/home/RobotMeQ/Configs/config_prd.ini
    docker cp /root/RobotMeQ/QuantData/live 5c239d668666:/home/RobotMeQ/QuantData/live
    docker cp 5c239d668666:/home/RobotMeQ/QuantData/trade_point_live /root/RobotMeQ/QuantData/trade_point_live

    老笔记本  激进策略  1031017763@qq.com
    docker start d63f10ba76df
    docker exec -it d63f10ba76df /bin/bash
    docker cp /root/RobotMeQ d63f10ba76df:/home/RobotMeQ
    docker cp /root/RobotMeQ/QuantData/live d63f10ba76df:/home/RobotMeQ/QuantData/live
    docker cp /root/RobotMeQ/Configs/config_prd.ini d63f10ba76df:/home/RobotMeQ/Configs/config_prd.ini
    docker cp /root/RobotMeQ/Run_prd_ETF_A.py d63f10ba76df:/home/RobotMeQ/Run_prd_ETF_A.py
    
    老台式机
    docker start 06acc8ba6062
    docker exec -it 06acc8ba6062 /bin/bash
    
--创建新项目    
    在conda的导航工具里新建环境，然后pycharm给项目选择需要的解释器，
    安装包时，conda的包不全，进入conda环境，conda activate robotme
        然后执行pip就行了，执行requirement也是进了conda的环境再执行
        requirement.txt要放在D:\anaconda3\condabin目录下
        进conda环境导出所有包执行：conda list -e > requirements.txt
        conda安装包：conda install --yes --file requirements.txt  
    
    由于没配conda的环境变量，要到anaconda安装目录的condabin下执行conda
    退出是conda deactivate
    
    我现在是两个环境并存，先是python安装了3.8，在c盘，配了全局代理，C:\ProgramData\pip\pip.ini
    给pycharm指定的解释器也是这个，用pycharm打开新的python项目时，会自动用virtual虚拟环境
    2023，0820装了anaconda，就把项目环境改成了anaconda的自己新建的env环境的python解释器。
    anaconda的多个python版本可以并存，我自己的python3.8也是和他们并存的
    只是有一点，我的python开了pip的代理，anaconda没有，不过anaconda包少，用conda安装包和进入conda的自己的环境，再用pip安装是一样的
    按理说pip.ini是通用的
    通过 pip config list 可以看到anaconda和全局用的是一个pip.ini，vpn的代理生效了。
    
    ---------------环境配好了，下面创建新项目----------------------
    用conda建环境后
    再到github创建项目，添加gitignore,拿到clone链接（别人的得拿，自己的项目不拿）
    本地用pycharm打开git点clone，放链接（自己地项目在github目录里直接选），选本地路径，导入后，改成conda的环境
    导入后放入自己的代码，点add，再commit，最后push
    
    """

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
