import baostock as bs
import pandas as pd
from RMQTool import Tools as RMTTools
import RMQData.Asset as RMQAsset


def getData_BaoStock(asset, start_date, end_date, bar_type):
    # 股票所有数据都能拿到，指数只有日线
    code = None
    # 股票：6开头sh，0或3开头sz
    # 指数：3开头是sz
    if asset.assetsCode.startswith('6'):
        code = 'sh.' + asset.assetsCode
    elif asset.assetsCode.startswith('0') or asset.assetsCode.startswith('3'):
        code = 'sz.' + asset.assetsCode
        if asset.assetsType == 'index':
            # 如果只根据名字判断，0,3以外全是sh，那指数0开头的代码只能加sh.前缀，才满足else，但文件名多了个sh.不好看，所以加这个assetsType
            # 当然影响不只是csv文件名，如果用mysql存，存表名是000001是股票，还是指数，还得区分，因此这个变量得加
            code = 'sh.' + asset.assetsCode

    lg = bs.login()
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 日周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    if asset.barEntity.timeLevel == 'd':
        # 日线数据，股票和指数接口一样
        rs = bs.query_history_k_data_plus(code,
                                          "date,open,high,low,close,volume",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")
    else:
        # 分钟先数据，只有股票
        """
        华尔街行情是不复权，通达信默认不复权，可以设置，
        东方财富默认前复权，我应该选前复权
        adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权
        前复权：以当前股价为基准向前复权计算股价。
        后复权：以上市首日股价作为基准向后复权计算股价
        前复权和后复权那哪种方式好
        各有优劣，如果是分析短周期数据，前后复权差别并不大；
        如果分析最近一段时间的数据，用前复权比较合适；
        如果是分析很长一段时间的数据，尤其是分析上市公司上市以来的所有数据，使用后复权比较合适。
        """
        rs = bs.query_history_k_data_plus(code,
                                          "date,time,open,high,low,close,volume",
                                          start_date=start_date, end_date=end_date,
                                          frequency=asset.barEntity.timeLevel, adjustflag="2")

    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    if 0 != len(data_list):
        if 'd' == asset.barEntity.timeLevel:
            result.loc[:, 'date'] = pd.to_datetime(result.loc[:, 'date'])
            result.rename(columns={'date': 'time'}, inplace=True)  # 为了和分钟级bar保持一致，修改列名为time
        else:
            # 证券宝的time字段20170703093500000，int类型处理不了，所以这里裁剪掉后三个0
            result.loc[:, 'time'] = [t[:-3] for t in result.loc[:, 'time']]
            result.loc[:, 'time'] = pd.to_datetime(result.loc[:, 'time'])
            result = result.loc[:, ['time', 'open', 'high', 'low', 'close', 'volume']]
    # 结果集输出到csv文件
    if 'backtest_bar' == bar_type:
        result.to_csv(asset.barEntity.backtest_bar, index=False)
    elif 'live_bar' == bar_type:
        # 实盘不管什么级别，只要最近的250bar就行
        windowDF = cut_by_bar_num(result, asset.barEntity.bar_num)
        windowDF.to_csv(asset.barEntity.live_bar, index=False)
    print(result)
    # 登出系统
    bs.logout()


def handle_TDX_data(asset):
    """
    把通达信导出的xls数据，另存为xlsx后，此函数将其处理为csv文件

    1、通达信软件行情页进入某个etf
    2、选择想要的级别
    3、点选项，数据导出，导出为xls  （只能选这个格式）
    4、把xls另存为xlsx
    """
    filePath = (RMTTools.read_config("RMQData_local", "tdx")
                + asset.assetsCode
                + '_'
                + asset.barEntity.timeLevel
                + '.xlsx')
    DataFrame = pd.read_excel(filePath)
    data = DataFrame.iloc[3:-1, 0:6]  # 含头不含尾，截取第3行到倒数第二行，第0列到第5列
    data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    # if time(11, 30) < datetime.now().time() < time(13):
    #     # 上午报错，中午重新加载数据时，要做特殊处理
    #     # 通达信中午收盘最后一个数据时间是13，改成11：30
    #     tempTime = data.iloc[-1, 0]
    #     tempTime2 = tempTime[0:12] + '11:30'
    #     data.iloc[-1, 0] = tempTime2

    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    # etf和指数分钟数据，够实盘用就行，回测用股票方便
    # 如果是回测数据：不用截断，写入到backtest_bar
    # windowDF = cut_by_bar_num(data, 250)
    data.to_csv(asset.barEntity.backtest_bar, columns=['open', 'high', 'low', 'close', 'volume'])


def cut_by_bar_num(df, bar_num):
    # 实盘只要bar_num条，最新的数据，够算指标就行，所以这里截断没用的旧数据
    length = len(df)
    if length >= bar_num:
        window = length - bar_num  # 起始下标比如是0~250，bar_num是250，iloc含头不含尾
        windowDF = df.iloc[window:length].copy()  # copy不改变原对象，不加copy会有改变临时对象的警告
        return windowDF
    else:
        # 数据不够bar_num条，就不用截取了
        return df


if __name__ == '__main__':
    """
    assetsType ： stock index ETF crypto
    ['5', '15', '30', '60', 'd']
    backtest_bar  live_bar
    """
    assetList = RMQAsset.asset_generator('600332', '', ['30'], 'stock', 1)
    for asset in assetList:
        # 接口取数据只能股票，回测方便
        # getData_BaoStock(asset, '2000-01-01', '2024-06-11', 'backtest_bar')
        # 日线要拿前250天的数据，单独加载，不然太慢
        getData_BaoStock(asset, '2002-01-01', '2024-06-30', 'backtest_bar')

        # 通达信拿到的数据，xlsx转为csv；主要实盘用，偶尔回测拿指数、ETF数据用
        # 如果是回测数据，handle_TDX_data末尾要改
        # handle_TDX_data(asset)