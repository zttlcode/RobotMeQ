from RMQData.Bar import Bar


class Stock(Bar):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType)


class Index(Bar):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType)


class ETF(Bar):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType)


class Crypto(Bar):
    def __init__(self, assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType):
        super().__init__(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType)
        self.back_test_bar_data = None  # 读取所有的回测数据
        self.back_test_cut_row = None  # 为了从读取的回测bar中，定位到自己要的回测开始时间


def asset_generator(assetsCode, assetsName, timeLevelList, assetsType):
    """
    :param assetsCode: 给代码
    :param assetsName: 给名字
    :param timeLevelList: 目前只支持5、15、30、60、d这几个级别
    :param assetsType: 资产类型  用于生成不同的资产
    :return: 给想要的级别，就能new对应级别的对象，放入列表
    """
    # 判断是否为多级别  多级别在实盘、生成新bar时，会不断用 当日最新成交量-累积成交量，算出当前bar的成交量
    isRunMultiLevel = False
    if len(timeLevelList) > 1:
        isRunMultiLevel = True
    # 根据级别列表，创建各级别资产对象
    assetList = []
    for timeLevel in timeLevelList:
        if assetsType == 'stock':
            assetList.append(Stock(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType))
        elif assetsType == 'index':
            assetList.append(Index(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType))
        elif assetsType == 'ETF':
            assetList.append(ETF(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType))
        elif assetsType == 'crypto':
            assetList.append(Crypto(assetsCode, assetsName, timeLevel, isRunMultiLevel, assetsType))
    return assetList
