"""
项目代号：本质
项目描述：炒股无非是看看各级别的k线、成交量，综合做决策，
本方法通过机器学习挖掘各级别之间的关系，分类当前交易点是否有效，并投票
实验用于回测，实盘用于实际交易

模型分类的，是此信号有效，无效，而不是买卖
有效就报信号，无效就不报
那我标注时，应该把所有交易点都留下，无效的标为无效，有效的标为有效，固定窗口
抓特征只抓上级
"""
import RMQStrategy.Strategy_nature as RMQStrategy

def back_test(asset):
    asset.barEntity.bar_DataFrame = pd.read_csv(bar_path, parse_dates=['time'])
            # 现在的策略是为了多级别实盘设计的，有很多细节
            # 但目前回测是单级别，因此对策略细节做出改动，选择复制个策略改动
            for asset.barEntity.bar_DataFrame
                # 把一个标的 一个级别的所有数据回测，记录交易点
                RMQStrategy.strategy(asset, strategy_result, IEMultiLevel)
            # 保存买卖点信息
            if asset.positionEntity.trade_point_list:  # 不为空，则保存
                df_tpl = pd.DataFrame(asset.positionEntity.trade_point_list)
                df_tpl.to_csv(RMTTools.read_config("RMQData", "trade_point_backtest")
                              + "trade_point_list_"
                              + asset.indicatorEntity.IE_assetsCode
                              + "_"
                              + asset.indicatorEntity.IE_timeLevel
                              + ".csv", index=False)
def filter1(asset):
    pd.read_csv(asset.positionEntity.trade_point_list path,)

def filter2(asset):
    pd.read_csv(asset.positionEntity.trade_point_list path,)


def run_experiment():
    """数据预处理"""
    # 1、数据获取  前复权
    baostock
    yfinance
    我的旧项目还得解耦一下
    问题只在run和strategy里，这俩都是为多级别设计的，
    run要写个新的，strategy里面加个新策略就好了，
        strategy_result和IEMultiLevel是多级别专属的，不应该成为通用，应改名为strategy_multiLevel,
        现在的run也是多级别专属的，应该改名为run_multiLevel,
        在Run.py里新加个run_singleLevel
        run和bar都是A股专属的，如果调港股，bar也要新加个

        策略最好还是分开文件，变动多，还是别动旧代码
    for asset in allAsset
        assetList = RMQAsset.asset_generator('000001','上证',['5', '15', '30', '60', 'd'],'index',1) # asset是code等信息
        for asset in assetList  # 每个标的所有级别
            back_test(asset)  # 2、传统策略回测，拿到交易点
            filter1(asset)  # 3、过滤交易点  交易对过滤法
            filter2(asset)  # 短线过滤方法
            """
            # 4、把预处理数据转为 单变量定长或变长分类，或 多变量定长或变长分类，
            # 变长是因为对于回归类策略，一对交易点的间隔可能很短，也可能很长，太长就加个最大限度
             # 组织数据
             依次遍历交易点，比如5分钟第一个交易点出现，此时拿到对应时间及label，按长度找到每个上级序列，加上label，还要沪深300
            """
            trans_data2label1(asset)
            trans_data2label2(asset)
            trans_data2label3(asset)
            trans_data2label4(asset)

            标注过交易点后，每个都要和buyandhold对比收益率，不应该出现效果差的

    """模型训练"""
    # 1、构建弱学习器模型
    建立model包，建个CNN.py
    先拿5分钟训练，数据进来，所有级别一起训练
    其他高级别训练放试验里


    # 2、构建集成学习模型
    """实验"""
    # 1、策略不变，过滤方式变
    # 2、策略不变，过滤方式不变，超参变
    # 3、策略变等等


def run_live():
    pass


if __name__ == '__main__':
    run_experiment()  # 实验回测
    # run_live()  # 实盘
