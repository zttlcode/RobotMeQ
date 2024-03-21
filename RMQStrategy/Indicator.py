import pandas as pd
from RMQTool import Tools as RMTTools


# 每个Asset对象 的父类Bar，都有一个指标对象，记录当前Asset对象的各种指标
class InicatorEntity:
    def __init__(self):
        self.bar_DataFrame = None  # 每个tick都会更新bar的dataframe
        self.tick_high = 0
        self.tick_low = 0
        self.tick_close = 0
        self.tick_time = None
        self.tick_volume = 0
        self.last_msg_time_1 = None  # 上次发消息时间，避免一个bar内，每个tick来都发信息，只发一次
        self.last_msg_time_2 = None  # 上次发消息时间，避免一个bar内，每个tick来都发信息，只发一次
        self.IE_assetsCode = None  # 复制资产代码
        self.IE_assetsName = None  # 复制资产名称
        self.IE_timeLevel = None  # 复制时间级别


# 一个Asset有多个时间级别，多个级别共用一个多级别指标，来交流指标信息
class InicatorEntityMultiLevel:
    def __init__(self):
        # EMA DIF DEA MACD MA_5 MA_10 MA_60 K D J RSI6 RSI12 RSI24
        self.level_5_EMA = None
        self.level_5_DIF = None
        self.level_5_DEA = None
        self.level_5_MACD = None
        self.level_5_MA_5 = None
        self.level_5_MA_10 = None
        self.level_5_MA_60 = None
        self.level_5_K = None
        self.level_5_D = None
        self.level_5_J = None
        self.level_5_RSI6 = None
        self.level_5_RSI12 = None
        self.level_5_RSI24 = None

        self.level_15_EMA = None
        self.level_15_DIF = None
        self.level_15_DEA = None
        self.level_15_MACD = None
        self.level_15_MA_5 = None
        self.level_15_MA_10 = None
        self.level_15_MA_60 = None
        self.level_15_K = None
        self.level_15_D = None
        self.level_15_J = None
        self.level_15_RSI6 = None
        self.level_15_RSI12 = None
        self.level_15_RSI24 = None

        self.level_30_EMA = None
        self.level_30_DIF = None
        self.level_30_DEA = None
        self.level_30_MACD = None
        self.level_30_MA_5 = None
        self.level_30_MA_10 = None
        self.level_30_MA_60 = None
        self.level_30_K = None
        self.level_30_D = None
        self.level_30_J = None
        self.level_30_RSI6 = None
        self.level_30_RSI12 = None
        self.level_30_RSI24 = None

        self.level_60_EMA = None
        self.level_60_DIF = None
        self.level_60_DEA = None
        self.level_60_MACD = None
        self.level_60_MA_5 = None
        self.level_60_MA_10 = None
        self.level_60_MA_60 = None
        self.level_60_K = None
        self.level_60_D = None
        self.level_60_J = None
        self.level_60_RSI6 = None
        self.level_60_RSI12 = None
        self.level_60_RSI24 = None

        self.level_day_EMA = None
        self.level_day_DIF = None
        self.level_day_DEA = None
        self.level_day_MACD = None
        self.level_day_MA_5 = None
        self.level_day_MA_10 = None
        self.level_day_MA_60 = None
        self.level_day_K = None
        self.level_day_D = None
        self.level_day_J = None
        self.level_day_RSI6 = None
        self.level_day_RSI12 = None
        self.level_day_RSI24 = None

    # 每个级别的Asset对象，每次进策略都会更新 多级别指标 的属性值
    def updateInicatorEntityMultiLevel(self, inicatorEntity, windowDF, DFLastRow):
        # 暂时只用到这一个指标，以后用到哪个，再加
        if inicatorEntity.IE_timeLevel == "5":
            self.level_5_K = windowDF.iloc[DFLastRow]['K']
        elif inicatorEntity.IE_timeLevel == "15":
            self.level_15_K = windowDF.iloc[DFLastRow]['K']
        elif inicatorEntity.IE_timeLevel == "30":
            self.level_30_K = windowDF.iloc[DFLastRow]['K']
        elif inicatorEntity.IE_timeLevel == "60":
            self.level_60_K = windowDF.iloc[DFLastRow]['K']
        elif inicatorEntity.IE_timeLevel == "d":
            self.level_day_K = windowDF.iloc[DFLastRow]['K']


def calEMA(df, term):
    # 第一个参数是数据，第二个参数是周期
    """计算移动平均值（即EMA）。
    12日EMA1的计算方式是：EMA（12）=前一日EMA（12）× 11/13＋今日收盘价× 2/13
    26日EMA2的计算方式是：EMA（26）=前一日EMA（26）× 25/27＋今日收盘价×2 /27
    """
    # 根据第二个参数term，计算快速（周期是12天）和慢速（周期是26天）的EMA值。
    for i in range(len(df)):
        if i == 0:  # 如果是第一天，则EMA值用当天的收盘价
            df.at[i, 'EMA'] = df.at[i, 'close']
            # 上面是通过df.at的形式访问索引行（比如第i行）和指定标签列（比如EMA列）的数值
            # at方法与之前loc以及iloc方法不同的是，at方法可以通过索引值和标签值访问
            # 而loc以及iloc方法只能通过索引值来访问
        if i > 0:  # 不是第一天，按上面公式计算当天的EMA值。
            df.at[i, 'EMA'] = (term - 1) / (term + 1) * df.at[i - 1, 'EMA'] + 2 / (term + 1) * df.at[i, 'close']
    # 计算完成后，把df的EMA列转换成列表类型的对象
    EMAList = list(df['EMA'])
    return EMAList


def calMACD(df, shortTerm=12, longTerm=26, DIFTerm=9):
    # 定义计算MACD的方法
    # 得到快速和慢速的EMA值
    shortEMA = calEMA(df, shortTerm)
    longEMA = calEMA(df, longTerm)
    # 计算MACD指标中的差离值（即DIF）
    # DIF =今日EMA（12）－今日EMA（26）
    # 注意，shortEMA和longEMA都是列表类型
    # 所以可以通过调用pd.Series方法把它们转换成Series类对象后再直接计算差值。
    df['DIF'] = pd.Series(shortEMA) - pd.Series(longEMA)
    for i in range(len(df)):
        if i == 0:  # 第一天
            df.at[i, 'DEA'] = df.at[i, 'DIF']  # ix可以通过标签名和索引来获取数据
        if i > 0:
            # 计算差离值的9日EMA（即MACD指标中的DEA）。用差离值计算它的9日EMA，这个值就是差离平均值（DEA）。
            # 今日DEA（MACD）=前一日DEA× 8/10＋今日DIF× 2/10
            df.at[i, 'DEA'] = (DIFTerm - 1) / (DIFTerm + 1) * df.at[i - 1, 'DEA'] + 2 / (DIFTerm + 1) * df.at[i, 'DIF']
    # 计算BAR柱状线。
    # BAR=2 × (DIF － DEA)这里乘以2的原因是，在不影响趋势的情况下，从数值上扩大DIF和DEA差值，这样观察效果就更加明显。
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    # 返回指定的列
    # return df[['time','DIF','DEA','MACD']]
    # 如果在后面的代码中还要用到df对象的其他列，则如下回df的全部列。
    return df


def calMA(df):
    maIntervalList = [5, 10, 60]
    # 虽然在后文中只用到了5日均线，但这里演示设置3种均线
    # 通过调用rolling方法，还是计算了3日、5日和10日均价，并把计算后的结果记录到当前行的MA_3、MA_5和MA_10这三列中
    # 这样做的目的是为了演示动态创建列的用法。
    for maInterval in maIntervalList:
        df['MA_' + str(maInterval)] = df['close'].rolling(window=maInterval).mean()
    return df


def calKDJ(df):
    """
    KDJ指标的波动与买卖信号有着紧密的关联，根据KDJ指标的不同取值，可以把这指标划分成三个区域：超买区、超卖区和观望区。
    一般而言，KDJ这三个值在20以下为超卖区，这是买入信号；这三个值在80以上为超买区，是卖出信号；如果这三个值在20到80之间则是在观望区。
    如果再仔细划分一下，当KDJ三个值在50附近波动时，表示多空双方的力量对比相对均衡，当三个值均大于50时，表示多方力量有优势，
    反之当三个值均小于50时，表示空方力量占优势。
    下面根据KDJ的取值以及波动情况，列出交易理论中比较常见的买卖策略。
    （1）KDJ指标中也有金叉和死叉的说法，即在低位K线上穿D线是金叉，是买入信号，反之在高位K线下穿D线则是死叉，是卖出信号。
    （2）一般来说，KDJ指标中的D线由向下趋势转变成向上是买入信号反之，由向上趋势变成向下则为卖出信号。
    （3）K的值进入到90以上为超买区，10以下为超卖区。对D而言，进入80以上为超买区，20以下为超卖区。
    此外，对K线和D线而言，数值50是多空均衡线。如果当前态势是多方市场，50是回档的支持线，即股价回探到KD值是50的状态时，
    可能会有一定的支撑，反之如果是空方市场，50是反弹的压力线，即股价上探到KD是50的状态时，可能会有一定的向下打压的压力。
    （4）一般来说，当J值大于100是卖出信号，如果小于10，则是买入信号。
    当然，上述策略仅针对KDJ指标而言，在现实的交易中，更应当从政策、消息、基本面和资金流等各个方面综合考虑。
    """
    # 计算KDJ
    # 把每一行（即每个交易日）的'MinLow'属性值设置为9天内收盘价（Low）的最小值。
    df['MinLow'] = df['low'].rolling(9, min_periods=9).min()
    # 填充NaN数据
    # 如果只执行上面那句，第1到第8个交易日的MinLow属性值将会是NaN，所以要通过下面这行代码，把这些交易日的MinLow属性值设置为
    # 9天内收盘价（Low）的最小值。
    # 如果要把修改后的数据写回到DataFrame中，必须加上inplace=True的参数
    df['MinLow'].fillna(value=df['low'].expanding().min(), inplace=True)
    # 把每个交易日的'MaxHigh'属性值设置为9天内的最高价
    df['MaxHigh'] = df['high'].rolling(9, min_periods=9).max()
    # 同样填充前8天的'MaxHigh'属性值
    df['MaxHigh'].fillna(value=df['high'].expanding().max(), inplace=True)
    # 根据算法计算每个交易日的RSV值
    # df['Close']等变量值是以列为单位，也就是说，在DataFrame中，可以直接以列为单位进行操作
    df['RSV'] = (df['close'] - df['MinLow']) / (df['MaxHigh'] - df['MinLow']) * 100
    # 通过for循环依次计算每个交易日的KDJ值
    for i in range(len(df)):
        if i == 0:  # 第一天
            df.at[i, 'K'] = 50
            df.at[i, 'D'] = 50
        if i > 0:
            df.at[i, 'K'] = df.at[i - 1, 'K'] * 2 / 3 + 1 / 3 * df.at[i, 'RSV']
            df.at[i, 'D'] = df.at[i - 1, 'D'] * 2 / 3 + 1 / 3 * df.at[i, 'K']
        df.at[i, 'J'] = 3 * df.at[i, 'K'] - 2 * df.at[i, 'D']
    return df


def calRSI(df):
    """
    一般来说，6日、12日和24日的RSI指标分别称为短期、中期和长期指标。和KDJ指标一样，RSI指标也有超买区和超卖区。
    具体而言，当RSI值在50到70之间波动时，表示当前属于强势状态，如继续上升，超过80时，则进入超买区，极可能在短期内转升为跌。
    反之RSI值在20到50之间时，说明当前市场处于相对弱势，如下降到20以下，则进入超卖区，股价可能出现反弹。

    先来讲述一下在实际操作中总结出来的RSI指标的缺陷。
    （1）周期较短（比如6日）的RSI指标比较灵敏，但快速震荡的次数较多，可靠性相对差些，而周期较长（比如24日）的RSI指标可靠性强，
    但灵敏度不够，经常会“滞后”的情况。
    （2）当数值在40到60之间波动时，往往参考价值不大，具体而言，当数值向上突破50临界点时，表示股价已转强，
    反之向下跌破50时则表示转弱。不过在实践过程中，经常会出现RSI跌破50后股价却不下跌，以及突破50后股价不涨。
    :param df:
    :param periodList:
    :return:
    """
    periodList = [6, 12, 24]  # 周期列表
    # 计算和上一个交易日收盘价的差值
    df['diff'] = df["close"] - df["close"].shift(1)
    # 由于第一行的diff值是NaN，因此需要用fillna方法把NaN值更新为0。
    df['diff'].fillna(0, inplace=True)

    # 在df对象中创建了up列，该列的值暂时和diff值相同，有正有负
    df['up'] = df['diff']
    # 过滤掉小于0的值。把up列中的负值设置成0，这样一来，up列中就只包含了“N日内收盘价的涨数”

    # 此链式调用有警告，改成下面的
    # df['up'][df['up'] < 0] = 0

    # 在Pandas中根据条件替换列中的值
    # df.loc[ df["column_name"] == "some_value", "column_name" ] = "value"
    # some_value = 需要被替换的值  value = 应该被放置的值。
    # 这里是想把up列，值小于0的替换为0
    df.loc[df['up'] < 0, 'up'] = 0

    # 用同样的方法，在df对象中创建了down列，并在其中存入了“N日内收盘价的跌数”
    df['down'] = df['diff']
    # 过滤掉大于0的值
    # df['down'][df['down'] > 0] = 0
    df.loc[df['down'] > 0, 'down'] = 0

    # 通过for循环，依次计算periodList中不同周期的RSI等值
    for period in periodList:
        # 算出了这个周期内收盘价涨数和的均值，并把这个均值存入df对象中的'upAvg'+str(period)列中
        df['upAvg' + str(period)] = df['up'].rolling(period).sum() / period
        df['upAvg' + str(period)].fillna(0, inplace=True)
        df['downAvg' + str(period)] = abs(df['down'].rolling(period).sum() / period)
        df['downAvg' + str(period)].fillna(0, inplace=True)
        df['RSI' + str(period)] = 100 - 100 / ((df['upAvg' + str(period)] / df['downAvg' + str(period)] + 1))
    return df


def calBolling(df):
    window_size = 20
    num_std = 2
    """"
    ·中心线 = N日移动均线
    ·上阻力线 = 中心线 + 两倍过去N天收盘价的标准差
    ·下支撑线 = 中心线 - 两倍过去N天收盘价的标准差

    如果过去20天收盘价波动比较大，那么布林带通道就比较宽，反之就比较狭窄。
    """
    # 计算均线和标准差  20日均线就是中轨
    df['rolling_mean'] = df['close'].rolling(window_size).mean()
    df['rolling_std'] = df['close'].rolling(window_size).std()

    # 计算布林通道
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * num_std)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * num_std)
    return df


def cal_DonchianChannel(df, window=20):
    # 要从第20条开始才能用
    # 计算上轨线
    df['donc_up'] = df['high'].rolling(window).max()
    # 计算下轨线
    df['donc_down'] = df['low'].rolling(window).min()
    # 计算唐奇安通道中心线
    df['donc_mid'] = (df['donc_up'] + df['donc_down']) / 2
    return df


if __name__ =='__main__':
    filePath = RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_000001_d.csv'
    DataFrame = pd.read_csv(filePath, encoding='gbk')
    df = cal_DonchianChannel(DataFrame)
    print(df)