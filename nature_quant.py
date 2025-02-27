import pandas as pd

import RMQData.Asset as RMQAsset
from RMQModel import Dataset as RMQDataset
from RMQModel import Evaluate as RMQEvaluate
from RMQModel import Label as RMQLabel
from RMQStrategy import Identify_market_types as RMQS_Identify_Market_Types
from RMQVisualized import Draw_Pyecharts as Draw_Pyecharts
import Run as Run


def pre_handle():
    """ """"""
    每个市场风格不同，混合训练会降低特色，
    A股数据，沪深300+中证500=A股前800家上市公司，港股、美股标普500、数字币市值前9
    数据来自证券宝，每个股票5种数据：日线、60m、30m、15m、5m。日线从该股发行日到2025年1月9日，分钟级最早为2019年1月2日。前复权，数据已压缩备份
    
    数据回测前，先判断下载的数据有没有非法值，find_zero_close_files
    
    A股 a800_stocks  a800_stocks_wait_handle_stocks
    row['code'][3:]  row['code_name']  '5', '15', '30', '60', 'd'  stock  1  A
    
    港股 hk_1000_stock_codes  hk_1000_stock_codes_wait_handle_stocks
    row['code']  row['name']  'd'  stock  1  HK
    
    美股 sp500_stock_codes  sp500_stock_codes_wait_handle_stocks
    row['code']  row['code']  'd'  stock  1  USA
    
    数字币 crypto_code  crypto_code_wait_handle_stocks
    row['code']  row['code']  '15', '60', '240', 'd'  crypto  1  crypto
    """
    allStockCode = pd.read_csv("./QuantData/asset_code/crypto_code.csv", dtype={'code': str})
    # Run.parallel_backTest(allStockCode)  # 回测，并行 需要手动改里面的策略名。
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'],
                                             row['code'],
                                             ['15', '60', '240', 'd'],
                                             'crypto',
                                             1, 'crypto')
        # 回测，保存交易点,加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
        # Run.run_back_test(assetList, "tea_radical_nature")  # 0:18:27.437876 旧回测，转tick，运行时长
        # Run.run_back_test_no_tick(assetList, "fuzzy_nature")  # 0:02:29.502122 新回测，不转tick
        # RMQS_Identify_Market_Types.run_backTest_label_market_condition(assetList)  # 回测标注日线级别行情类型 该上面时间级别为d

        # 各级别交易点拼接在一起
        # concat_trade_point(assetList, "tea_radical_nature")
        """
        过滤交易点
            strategy_name: identify_Market_Types 
                            tea_radical_nature  
                            fuzzy_nature    回测后，标注前，一定要process_fuzzy_trade_point_csv()预处理
                            c4_trend_nature
                            c4_oscillation_boll_nature
                            c4_oscillation_kdj_nature
                            c4_breakout_nature
                            c4_reversal_nature
                            extremum
            label_name: 
                label1: 多级别交易点合并，校验交易后日线级别涨跌幅、40个bar内趋势 tea_radical_nature的是concat，其他都是单级别
                label2：单级别校验各自涨跌幅、40个bar内趋势
                label3：单级别校验各自MACD、DIF是否维持趋势
                label4：单级别校验各自MACD、DIF+40个bar内趋势
        """
        # RMQLabel.label(assetList, "fuzzy_nature", "label1")
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
        # Draw_Pyecharts.show(assetList, "single", "fuzzy_nature", "_label1")
        """
        计算收益率
            is_concat: True 计算合并交易点的收益率  此时flag只会是 _concat 或 _concat_label1
                       False 计算各个级别，此时flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                        fuzzy的各级别flag也有 _label1
            strategy_name : tea_radical_nature  fuzzy_nature
                            c4_trend_nature
                            c4_oscillation_boll_nature
                            c4_oscillation_kdj_nature
                            c4_breakout_nature
                            c4_reversal_nature
        """
        # RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_boll_nature", False, True)


def run_experiment():
    """"""
    """
    标注完成，准备训练数据
    由于两个数据集都要做，因此写俩方法串行，别删
        按照原始标注方法，_TRAIN 最多24.6万  _TEST 最多14.6万
        limit_length==0 代表不截断，全数据
        
        flag: _TRAIN 训练集截前500个股票，  _TEST 测试集截后300个  截之前按固定随机数乱序了
        time_point_step: 截取的时间步，最长500，最少得是8以上，因为很多时序模型需要得序列长度最少是8
        limit_length：限制长度是为了方便debug时调试，数据太多加载太慢
        handle_uneven_samples: macd策略样本不均，其他策略不一定有这个问题，所以这里控制要不要处理
        strategy_name: 为了读回测点文件，     
                        identify_Market_Types  不需要处理样本不均
                        fuzzy_nature
                        tea_radical_nature
                        c4_trend_nature
                        c4_oscillation_boll_nature
                        c4_oscillation_kdj_nature
                        c4_breakout_nature
                        c4_reversal_nature
                        extremum  不需要处理样本不均
        feature_plan_name: 不同特征组织方案
                feature_tea_concat 日线、小时线、指数日线
                feature_tea_multi_level macd5分钟、15分钟、30分钟
                feature_fuzzy_multi_level
                feature_all
                feature_tea_macd
                feature_c4_trend
                feature_c4_oscillation_boll
                feature_c4_oscillation_kdj
                feature_c4_breakout
                feature_c4_reversal  
                feature_extremum
        p2t_name: point_to_ts1 单级别
                  point_to_ts2 多级别
        label_name: 合并标注交易点的  此时flag只会是 _concat_label1
                    各级别标注交易点  "_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                    fuzzy的各级别flag也有 _label1
        name: 标的_级别_行数 ts文件命名，跟limit_length对应，这文件有多少条数据
                跑单级别时，在Dataset里只填对应级别        
    """
    RMQDataset.prepare_dataset("_TRAIN", "A_15_2w", 160,
                               20000, True,
                               "c4_reversal_nature", "feature_c4_reversal",
                               "point_to_ts1", "_label1")
    RMQDataset.prepare_dataset("_TEST", "A_15_2w", 160,
                               10000, True,
                               "c4_reversal_nature", "feature_c4_reversal",
                               "point_to_ts1", "_label1")


def run_live():
    RMQDataset.prepare_dataset_single("_TEST", "2w_c4_oscillation_kdj_nature_20", 20,
                                      20000, True,
                                      "c4_oscillation_kdj_nature", "feature_c4_oscillation_kdj",
                                      "point_to_ts1", "_label1", 3)
    """
603737 结束 310
603786 结束 170
    """
    assetList = RMQAsset.asset_generator('603786',
                                         '',
                                         ['d'],
                                         'stock',
                                         1, 'A')
    RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_kdj_nature",
                            False, True)
    # 读取CSV文件
    RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_kdj_nature",
                            True, True)


if __name__ == '__main__':
    # pre_handle()  # 数据预处理
    run_experiment()  # 所有股票组成训练集
    # run_live()  # 单独推理一个股票
    pass
