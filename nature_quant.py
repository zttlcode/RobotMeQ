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
from RMQModel import Dataset as RMQDataset
from RMQModel import Evaluate as RMQEvaluate
from RMQVisualized import Draw_Pyecharts as RMQDraw_Pyecharts
from RMQModel import Label as RMQLabel
from RMQModel import Identify_Market_Types as RMQM_Identify_Market_Types
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
    a800_stocks
    a800_wait_handle_stocks
    """
    allStockCode = pd.read_csv("./QuantData/a800_stocks.csv")
    # 回测，并行 需要手动改里面的策略名
    Run.parallel_backTest(allStockCode)
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['d'],
                                             'stock',
                                             1, 'A')
        """
        回测，保存交易点
        加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
        strategy_name : tea_radical_nature  fuzzy_nature
        """
        # Run.run_back_test(assetList, "tea_radical_nature")  # 0:18:27.437876 旧回测，转tick，运行时长
        # Run.run_back_test_no_tick(assetList, "fuzzy_nature")  # 0:02:29.502122 新回测，不转tick
        # RMQM_Identify_Market_Types.run_backTest_label_market_condition(assetList)  # 回测标注日线级别行情类型 该上面时间级别为d

        # 各级别交易点拼接在一起
        # concat_trade_point(assetList, "tea_radical_nature")

        """
        过滤交易点
            strategy_name: tea_radical_nature  fuzzy_nature  identify_Market_Types
            label_name: 
                label1: 多级别交易点合并，校验交易后日线级别涨跌幅、40个bar内趋势
                label2：单级别校验各自涨跌幅、40个bar内趋势
                label3：单级别校验各自MACD、DIF是否维持趋势
                label4：单级别校验各自MACD、DIF+40个bar内趋势
        """
        # RMQLabel.label(assetList, "identify_Market_Types", "label1")

        """
        画K线买卖点图
            method_name:
                mix: 自己在函数里自定义，用什么级别组合自己改，不需要flag
                multi_concat：多级别点位合并图，此时flag只会是 _concat 或 _concat_label1
                single：单级别图，会用到不同过滤方式，因此flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                fuzzy的各级别flag也有 _label1
            strategy_name: tea_radical_nature  fuzzy_nature
        """
        # RMQDraw_Pyecharts.show(assetList, "single", "fuzzy_nature", "_label1")

        """
        计算收益率
            is_concat: True 计算合并交易点的收益率  此时flag只会是 _concat 或 _concat_label1
                       False 计算各个级别，此时flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                        fuzzy的各级别flag也有 _label1
            strategy_name : tea_radical_nature  fuzzy_nature
        """
        # RMQEvaluate.return_rate(assetList, False, "_label1", "fuzzy_nature")

    """
    标注完成，准备训练数据
    由于两个数据集都要做，因此写俩方法串行，别删
        按照原始标注方法，_TRAIN 最多24.6万  _TEST 最多14.6万
        limit_length==0 代表不截断，全数据
        flag: _TRAIN 训练集截前500个股票，  _TEST 测试集截后300个  截之前按固定随机数乱序了
        name: 给ts文件命名，跟limit_length对应，这文件有多少条数据
        time_point_step: 截取的时间步，最长500，最少得是8以上，因为很多时序模型需要得序列长度最少是8
        limit_length：限制长度是为了方便debug时调试，数据太多加载太慢
        handle_uneven_samples: macd策略样本不均，其他策略不一定有这个问题，所以这里控制要不要处理
        strategy_name: 为了读回测点文件，tea_radical_nature  fuzzy_nature  identify_Market_Types
        feature_plan_name: 不同特征组织方案
                feature1 日线、小时线、指数日线
                feature2 macd5分钟、15分钟、30分钟
        p2t_name: 针对不同特征，截不同得训练集数据
                point_to_ts1 针对feature1截收盘价和成交量
                point_to_ts2 针对feature2截MACD、KDJ、close
        label_name: 合并标注交易点的  此时flag只会是 _concat_label1
                    各级别标注交易点  "_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                    fuzzy的各级别flag也有 _label1
    """
    # RMQDataset.prepare_dataset("_TRAIN", "2w_identify_Market_Types_20", 20, 200000, False,
    #                            "identify_Market_Types", "feature5", "point_to_ts1",
    #                            "_label1")
    # RMQDataset.prepare_dataset("_TEST", "2w_identify_Market_Types_20", 20, 100000, False,
    #                            "identify_Market_Types", "feature5", "point_to_ts1",
    #                            "_label1")



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
