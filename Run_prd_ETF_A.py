import requests
import pandas as pd
import numpy as np
from time import sleep
from datetime import datetime, time
from multiprocessing import Process
import sys

sys.path.append("/home/RobotMeQ")

import RMQData.Tick as RMQTick
import RMQStrategy.Strategy as RMQStrategy
import RMQStrategy.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset
import RMQData.Bar_HistoryData as RMQBar_HistoryData
from RMQTool import Tools as RMTTools


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
    processes = [Process(target=run_live,
                         args=(RMQAsset.asset_generator('510050', '上证50', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159915', '创业板', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('510300', '沪深300指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('510500', '中证500指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512100', '中证1000指数', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('588000', '科创50', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159920', '恒生', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159941', '纳指', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512690', '酒', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512480', '半导体', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515030', '新能源车', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513050', '中概互联', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513060', '恒生医疗', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515790', '光伏', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('516970', '基建', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512660', '军工', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159611', '电力', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512200', '地产', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512170', '医疗', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512800', '银行', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512980', '传媒', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('512880', '证券', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515220', '煤炭', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159766', '旅游', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159865', '养殖', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('518880', '黄金', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159985', '豆粕', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159980', '有色', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159996', '家电', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159819', '人工智能', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159869', '游戏', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('515880', '通信', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('516150', '稀土', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('516110', '汽车', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('159866', '日经', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513360', '教育', ['5', '15', '30', '60', 'd'], 'ETF'),)),
                 Process(target=run_live,
                         args=(RMQAsset.asset_generator('513030', '德国', ['5', '15', '30', '60', 'd'], 'ETF'),))
                 ]

    for p in processes:
        # 启动进程
        p.start()

    for p in processes:
        p.join()
        p.close()


if __name__ == '__main__':

    """
--部署代码
    一个新服务器，安装docker : yum update更新一下库
    然后执行 curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    执行 service docker start 启动服务      systemctl enable docker 开机启动
    把RobotMeQ放到服务器~目录下，运行 docker build -t python:3.8.2 .
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
    
    docker start 26dbf8178821
    docker exec -it 26dbf8178821 /bin/bash
    nohup python -u /home/RobotMeQ/Run_prd_ETF_A.py >> /home/log.out 2>&1 &
    tail -f /home/log.out
    docker cp /root/RobotMeQ 26dbf8178821:/home/RobotMeQ
    docker cp /root/RobotMeQ/Run_prd_ETF_A.py 26dbf8178821:/home/RobotMeQ/Run_prd_ETF_A.py
    docker cp /root/RobotMeQ/requirements.txt 26dbf8178821:/home/RobotMeQ/requirements.txt
    docker cp /root/RobotMeQ/QuantData/live 26dbf8178821:/home/RobotMeQ/QuantData/live2
    
--创建新项目    
    在conda的导航工具里新建环境，然后pycharm给项目选择需要的解释器，
    安装包时，conda的包不全，进入conda环境，conda activate robotme
        然后执行pip就行了，执行requirement也是进了conda的环境再执行
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
