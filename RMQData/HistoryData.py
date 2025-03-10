import baostock as bs
import pandas as pd
from RMQTool import Tools as RMTTools
import RMQData.Asset as RMQAsset
import akshare as ak


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
                                          frequency="d", adjustflag="3")
    else:
        # 分钟先数据，只有股票
        """
        华尔街行情是不复权，通达信默认不复权，可以设置，
        东方财富默认前复权，我应该选前复权  2025 0224 前复权会导致早期价格出现负数，后复权也有可能，因此都改为不复权
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
                                          frequency=asset.barEntity.timeLevel, adjustflag="3")

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
    else:
        windowDF = cut_by_bar_num(result, 270)
        # 登出系统
        bs.logout()
        return windowDF
    print(result)
    # 登出系统
    bs.logout()


def query_hs300_stocks():
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取沪深300成分股
    rs = bs.query_hs300_stocks()
    print('query_hs300 error_code:' + rs.error_code)
    print('query_hs300  error_msg:' + rs.error_msg)

    # 打印结果集
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        hs300_stocks.append(rs.get_row_data())
    result = pd.DataFrame(hs300_stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("../QuantData/asset_code/a_hs300_stocks.csv", encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def query_zz500_stocks():
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取中证500成分股
    rs = bs.query_zz500_stocks()
    print('query_zz500 error_code:' + rs.error_code)
    print('query_zz500  error_msg:' + rs.error_msg)

    # 打印结果集
    zz500_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        zz500_stocks.append(rs.get_row_data())
    result = pd.DataFrame(zz500_stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("../QuantData/asset_code/a_zz500_stocks.csv", encoding="utf-8", index=False)
    print(result)

    # 登出系统
    bs.logout()


def query_ipo_date(stock_code):
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取证券基本资料
    rs = bs.query_stock_basic(code=stock_code)
    # rs = bs.query_stock_basic(code_name="浦发银行")  # 支持模糊查询
    print('query_stock_basic respond error_code:' + rs.error_code)
    print('query_stock_basic respond  error_msg:' + rs.error_msg)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 登出系统
    bs.logout()
    return str(result.at[0, 'ipoDate'])


def get_ipo_date_for_stock():
    # 读取 CSV 文件，得到 DataFrame 对象
    file_path = "../QuantData/asset_code/a_zz500_stocks.csv"  # 文件路径
    df = pd.read_csv(file_path)
    # 遍历每行数据，调用 query_ipo_date 并修改第一列的日期
    for index, row in df.iterrows():
        stock_code = row['code']  # 提取编号
        new_date = query_ipo_date(stock_code)  # 查询得到新的日期
        df.at[index, 'updateDate'] = new_date  # 覆盖第一列的日期
    df.rename(columns={'updateDate': 'ipodate'}, inplace=True)
    # 将修改后的 DataFrame 写回 CSV 文件
    df.to_csv(file_path, index=False)


def get_stock_from_code_csv():
    allStockCode = pd.read_csv("../QuantData/asset_code/a_zz500_stocks.csv")
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:], row['code_name'], ['5', '15', '30', '60', 'd'],
                                             'stock', 1, 'A')  # asset是code等信息
        for asset in assetList:  # 每个标的所有级别
            lg = bs.login()
            print('login respond error_code:' + lg.error_code)
            print('login respond  error_msg:' + lg.error_msg)
            if asset.barEntity.timeLevel == 'd':
                # 日线数据，股票和指数接口一样
                rs = bs.query_history_k_data_plus(row['code'],
                                                  "date,open,high,low,close,volume",
                                                  start_date=row['ipodate'],
                                                  frequency="d", adjustflag="3")
            else:
                # 分钟先数据，只有股票
                rs = bs.query_history_k_data_plus(row['code'],
                                                  "date,time,open,high,low,close,volume",
                                                  start_date=row['ipodate'],
                                                  frequency=asset.barEntity.timeLevel, adjustflag="3")

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
            result.to_csv(asset.barEntity.backtest_bar, index=False)
            print(result)
            # 登出系统
            bs.logout()


def handle_TDX_data(asset, is_live):
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
    if is_live:
        windowDF = cut_by_bar_num(data, asset.barEntity.bar_num)
        windowDF.to_csv(asset.barEntity.live_bar, columns=['open', 'high', 'low', 'close', 'volume'])
    else:
        # 如果是回测数据：不用截断，写入到backtest_bar
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


def get_sp500_code():
    # 从Wikipedia获取标普500成分股列表
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]

    # 提取股票代码（Symbol列）
    stock_codes = df['Symbol']
    # 修改列名，将"date"改为"time"
    stock_codes.rename(columns={'Symbol': 'code'}, inplace=True)
    # 将股票代码保存到CSV文件
    stock_codes.to_csv('../QuantData/asset_code/sp500_stock_codes.csv', index=False)

    print("CSV文件已保存。")


def get_sp500_data():
    # 读取s&p500_stock_codes.csv文件，获取股票代码
    df = pd.read_csv('../QuantData/asset_code/sp500_stock_codes.csv')
    stock_codes = df['code'].tolist()

    # 循环遍历每个股票代码
    for symbol in stock_codes:
        try:
            # 获取股票历史数据
            stock_data = ak.stock_us_daily(symbol=symbol, adjust="")  # 调整方式根据需要可以修改

            # 修改列名，将"date"改为"time"
            stock_data.rename(columns={'date': 'time'}, inplace=True)

            # 保存数据为CSV文件，文件名为股票代码
            stock_data.to_csv(f'D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_USA_{symbol}_d.csv', index=False)

            print(f"{symbol} 数据已保存为 {symbol}.csv")
        except Exception as e:
            print(f"无法获取 {symbol} 的数据: {e}")


def get_hk_stock_code():
    # 获取港股知名股，110多家
    # stock_hk_famous_spot_em_df = ak.stock_hk_famous_spot_em()
    # stock_hk_famous_spot_em_df_filtered = stock_hk_famous_spot_em_df[['代码', '名称']]
    # stock_hk_famous_spot_em_df_filtered.to_csv('../QuantData/hk_famous.csv', index=False)

    # 获取港股所有股票
    # stock_hk_spot_em_df = ak.stock_hk_spot_em()
    # stock_hk_spot_em_df_filtered = stock_hk_spot_em_df[['代码', '名称']]
    # stock_hk_spot_em_df_filtered.to_csv('../QuantData/hk_all_stocks.csv', index=False)

    df1 = pd.read_csv('../QuantData/asset_code/hk_all_stocks.csv', dtype={'代码': str})
    df2 = pd.read_csv('../QuantData/asset_code/hk_famous.csv', dtype={'代码': str})

    # 获取df1的前1000行  这个不是按市值排的，我在网上找不到按市值排名的数据
    df1_top_1000 = df1.head(1000)

    # 合并df1和df2
    df_combined = pd.concat([df1_top_1000, df2], ignore_index=True)

    # 重命名列名
    df_combined.rename(columns={'代码': 'code', '名称': 'name'}, inplace=True)

    # 按code列去重
    df_combined_unique = df_combined.drop_duplicates(subset=['code'])

    # 保存为新的CSV文件
    df_combined_unique.to_csv('../QuantData/asset_code/hk_1000_stock_codes.csv', index=False)


def get_hk_stock_data():
    # 读取s&p500_stock_codes.csv文件，获取股票代码
    df = pd.read_csv('../QuantData/asset_code/hk_1000_stock_codes.csv', dtype={'code': str})

    stock_codes = df['code'].tolist()

    # 循环遍历每个股票代码
    for symbol in stock_codes:
        try:
            # 获取股票历史数据
            stock_data = ak.stock_hk_daily(symbol=symbol, adjust="")  # 调整方式根据需要可以修改

            # 修改列名，将"date"改为"time"
            stock_data.rename(columns={'date': 'time'}, inplace=True)

            # 保存数据为CSV文件，文件名为股票代码
            stock_data.to_csv(f'D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_HK_{symbol}_d.csv', index=False)

            print(f"{symbol} 数据已保存为 {symbol}.csv")
        except Exception as e:
            print(f"无法获取 {symbol} 的数据: {e}")


if __name__ == '__main__':
    """
    stock index ETF crypto
    ['5', '15', '30', '60', 'd']
    backtest_bar  live_bar
    """
    assetList = RMQAsset.asset_generator('600332', '', ['5', '15', '30', '60', 'd'],
                                         'stock', 1, 'A')
    for asset in assetList:
        # 接口取数据只能股票，回测方便
        # getData_BaoStock(asset, '2000-01-01', '2024-06-11', 'backtest_bar')
        pass
    # 获取股票代码、获取股票发行日、获取股票各级别数据
    # get_stock_from_code_csv()  # 日线能从发行日开始，分钟级别最早是2019年元旦
    pass


