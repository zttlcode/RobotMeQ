"""
项目代号：本质
项目描述：炒股无非是看看各级别的k线、成交量，综合做决策，
本方法通过机器学习挖掘各级别之间的关系，分类当前交易点是否有效，并投票
实验用于回测，实盘用于实际交易

模型分类的，是此信号有效，无效，而不是买卖  有效为1，失效为0
有效就报信号，无效就不报
我的方法也抽取了某个时间窗口的特征，看现在的价格走势满足过去哪些特征，以此特征做识别，识别以前遇到这种情况指标会不会失效。  ！！！！！！！！

那我标注时，应该把所有交易点都留下，无效的标为无效，有效的标为有效，固定窗口
抓特征只抓上级

多种指标，就是多个特征，需要捕获特征之间的相关性么？像3论文大纲的 itransformer一样？
patchtst  分段？通道独立我可以试试，时间段token，每个特征各自进transformer，我模仿一下，每个特征各自进cnn？
"""
import pandas as pd

import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools
from RMQTool import Yield as RMQYield
from RMQModel import Dataset as RMQDataset
from RMQModel import Label as RMQLabel
from RMQVisualized import Draw_Pyecharts as RMQDraw_Pyecharts
import Run as Run


def pre_handle():
    """ """"""
    数据预处理
    A股、港股、美股、数字币。 每个市场风格不同，混合训练会降低特色，
        目前只用A股数据，沪深300+中证500=A股前800家上市公司
        涉及代码：HistoryData.py新增query_hs300_stocks、query_hs500_stocks，获取股票代码，
                get_ipo_date_for_stock、query_ipo_date 找出股票代码对应发行日期
                get_stock_from_code_csv 通过股票代码、发行日期，获取股票各级别历史行情
        待实验市场：港股、美股标普500、数字币市值前10
        数据来自证券宝，每个股票5种数据：日线、60m、30m、15m、5m。日线从该股发行日到2025年1月9日，分钟级最早为2019年1月2日。前复权，数据已压缩备份
    所有数据进行单级别回测，保留策略交易点，多进程运行
        目前策略：MACD+KDJ  （回归）
        涉及代码：旧代码在Run.py，5分钟bar转tick，给多级别同时用，回测一个股票5年要3小时。
                为提高效率，单级别运行，启动10线程，2台电脑，预计2、3天跑完4000个行情
        待实验策略：王立新ride-moon （趋势）  
                布林
                均线等，看是否比单纯指标有收益率提升
                （第三种方法、提前5天，直接抽特征自己发信号，不用判断当前信号是否有效）
    """
    allStockCode = pd.read_csv("./QuantData/a800_wait_handle_stocks.csv")
    # 回测，并行 需要手动改里面的策略名
    # Run.parallel_backTest(allStockCode)
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['5', '15', '30', '60', 'd'],
                                             'stock',
                                             1, 'A')

        # 回测，保存交易点
        # 加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
        # Run.run_back_test(assetList, "tea_radical_nature")  # 0:18:27.437876 旧回测，转tick，运行时长
        # Run.run_back_test_no_tick(assetList, "tea_radical_nature")  # 0:02:29.502122 新回测，不转tick
        # 各级别交易点拼接在一起
        # concat_trade_point(assetList, "tea_radical_nature")
        # 过滤交易点1
        # RMQLabel.tea_radical_filter1(assetList, "tea_radical_nature")
        # 计算收益率  _5 _15 _30 _60 _d _concat _concat_labeled
        # RMQYield.cal_return_rate(assetList[0], "_concat_filter1", "tea_radical_nature")
        for asset in assetList:
            # 过滤交易点2
            # RMQLabel.tea_radical_filter2(asset, "tea_radical_nature")
            # 过滤交易点3
            # RMQLabel.tea_radical_filter3(asset, "tea_radical_nature")
            # 过滤交易点4
            # RMQLabel.tea_radical_filter4(asset, "tea_radical_nature")

            flag0 = "_" + asset.barEntity.timeLevel  # 原始交易点
            flag1 = "_concat_filter1"  # _concat _concat_filter1  多级别组合+标注交易点
            flag2 = "_" + asset.barEntity.timeLevel + "_filter3"  # 各级别标注交易点

            # 计算收益率
            # RMQYield.cal_return_rate(asset, flag2, "tea_radical_nature")
            # 画点位图
            RMQDraw_Pyecharts.show_single(asset, flag2)  # 1、生成单级别图

        # 画点位图
        RMQDraw_Pyecharts.show_multi_concat(assetList, "_concat_filter1")  # 2、多级别混合图
        RMQDraw_Pyecharts.show_mix(assetList)  # 3、自己在函数里自定义

        # print(assetList[0].assetsCode + "标注完成")
    # 过滤交易点完成，准备训练数据
    """
    增加标识——是否处理样本不均
    tea策略买入点太多，filter1过滤后也是样本不均，导致大量无效买入
    我在损失函数层面实验了cost-sensitive，从
        criterion = nn.CrossEntropyLoss() 改为
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.59, 0.08, 0.08]))  没什么用
    https://zhuanlan.zhihu.com/p/494220661  
        这篇提到了其他解决办法：
            模型层面用决策树、
            集成学习中把少的样本重复抽样，组成训练子集，给单个模型
            样本极端少只有几十个时，将分类问题考虑成异常检测
        这实验起来有些麻烦，我先尝试直接删样本吧，handle_uneven_samples True处理，False不处理，按4类中最少的为准，删除其他样本
    """
    # 最多24.6万  limit_length==0 代表不截断，全数据
    # RMQDataset.prepare_dataset("_TRAIN", "2w", 250, 20000, True, "tea_radical_nature")
    # 最多14.6万
    # RMQDataset.prepare_dataset("_TEST", "2w", 250, 10000, True, "tea_radical_nature")


def run_experiment():
    """模型训练"""
    # 1、构建弱学习器模型
    """
    建立model包，建个CNN.py
    先拿5分钟训练，数据进来，所有级别一起训练
    其他高级别训练放试验里
    """

    # 2、构建集成学习模型
    """实验"""
    # 1、策略不变，过滤方式变
    # 2、策略不变，过滤方式不变，超参变
    # 3、策略变等等


def run_live():
    pass


if __name__ == '__main__':
    pre_handle()  # 数据预处理
    # run_experiment()  # 实验回测
    # run_live()  # 实盘
