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

import RMQStrategy.Strategy_nature as RMQStrategy
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools


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
    df = pd.read_csv("所有股票代码.csv")
    """ """"""
    过滤交易点
    
    读取所有交易点位，和他对应的行情数据，判断这个点位出现后，他后面的行情是怎么样的？
    """
    for index, row in df.iterrows():
        assetList = RMQAsset.asset_generator(row['code'], '上证', ['5', '15', '30', '60', 'd'], 'stock',
                                             1)  # asset是code等信息
        for asset in assetList:  # 每个标的所有级别
            back_test(asset)  # 2、传统策略回测，拿到交易点
            # filter1(asset)  # 3、过滤交易点  交易对过滤法
            # filter2(asset)  # 短线过滤方法
            """
            # 4、把预处理数据转为 单变量定长或变长分类，或 多变量定长或变长分类，
            # 变长是因为对于回归类策略，一对交易点的间隔可能很短，也可能很长，太长就加个最大限度
             # 组织数据
             依次遍历交易点，比如5分钟第一个交易点出现，此时拿到对应时间及label，按长度找到每个上级序列，加上label，还要沪深300
            """
            # 把有效交易点和原视数据结合，标注有效、无效
            # trans_point2label(asset)

def filter1(asset):
    pd.read_csv(asset.positionEntity.trade_point_list.path)


def filter2(asset):
    pd.read_csv(asset.positionEntity.trade_point_list.path)


def trans_point2label():
    pass
    # 标注过交易点后，每个都要和buyandhold对比收益率，不应该出现效果差的


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
    pre_handle() # 数据预处理
    # run_experiment()  # 实验回测
    # run_live()  # 实盘
