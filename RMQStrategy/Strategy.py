import RMQStrategy.Position as RMQPosition
import RMQStrategy.Indicator as RMQIndicator
import RMQStrategy.Indicator_analyse as RMQAnalyse
from RMQTool import Message


class StrategyResultEntity:
    def __init__(self):
        self.strategy_t = None
        self.live = True
        self.msg_level_5 = "无"
        self.msg_level_15 = "无"
        self.msg_level_30 = "无"
        self.msg_level_60 = "无"
        self.msg_level_day = "无"
        self.msg_level_week = None

    # 组合多级别策略结果，拼接发送交易建议
    def editMsg(self, inicatorEntity, post_msg):
        if self.live:
            # 设置消息
            if inicatorEntity.IE_timeLevel == "5":
                self.msg_level_5 = post_msg
            elif inicatorEntity.IE_timeLevel == "15":
                self.msg_level_15 = post_msg
            elif inicatorEntity.IE_timeLevel == "30":
                self.msg_level_30 = post_msg
            elif inicatorEntity.IE_timeLevel == "60":
                self.msg_level_60 = post_msg
            elif inicatorEntity.IE_timeLevel == "d":
                self.msg_level_day = post_msg
            # 编辑标题
            # title_time = datetime.now().strftime('%Y-%m-%d') + '操作'
            title = "图表派多级别策略"
            # 组织消息
            # mail_msg = Message.build_msg_HTML(title, self)
            mail_msg = Message.build_text(self)
            # 需要异步发送，此函数还没写，暂时先同步发送
            # res = Message.PostQQmail(mail_msg)
            res = Message.Postfeishu(mail_msg)
            if res:
                print('发送成功')
            else:
                print('发送失败')

    # 综合判断多个策略结果
    def synthetic_judge(self):
        pass


def strategy(positionEntity, inicatorEntity, bar_num, strategy_result, IEMultiLevel):
    # 1、每次tick重新计算指标数据
    # EMA DIF DEA MACD MA_5 MA_10 MA_60 K D J RSI6 RSI12 RSI24
    # start = datetime.now()

    # inicatorEntity.bar_DataFrame = RMQIndicator.calMA(inicatorEntity.bar_DataFrame)
    # inicatorEntity.bar_DataFrame = RMQIndicator.calMACD(inicatorEntity.bar_DataFrame)
    # inicatorEntity.bar_DataFrame = RMQIndicator.calKDJ(inicatorEntity.bar_DataFrame)
    # inicatorEntity.bar_DataFrame = RMQIndicator.calRSI(inicatorEntity.bar_DataFrame)
    # 耗时样例：end 0:00:00.109344 1014  耗时随规模线性增长，时间复杂度为O（n）

    # 指标计算为了省时间，设置个固定数据量的时间窗口
    length = len(inicatorEntity.bar_DataFrame)
    window = length - bar_num  # 起始下标比如是0~60，100~160等，bar_num是60，iloc含头不含尾。现在bar_num是250
    windowDF = inicatorEntity.bar_DataFrame.iloc[window:length].copy()  # copy不改变原对象，不加copy会有改变临时对象的警告
    windowDF = windowDF.reset_index(drop=True)  # 重置索引，这样df索引总是0~59

    # windowDF = RMQIndicator.calMA(windowDF)
    # windowDF = RMQIndicator.calMACD(windowDF)
    # windowDF = RMQIndicator.calKDJ(windowDF)
    # windowDF = RMQIndicator.calRSI(windowDF)
    # 耗时样例：end 0:00:00.015624 1449  耗时不随规模增长，一直保持在15毫秒，时间复杂度为O（1）
    # print("end", datetime.now()-start, len(inicatorEntity.bar_DataFrame), len(windowDF))
    # 每个tick都要重新算，特别耗时  时间窗口对性能提升很大 用windowDF，不要用上面的 inicatorEntity.bar_DataFrame

    # 2、策略主方法，再此更换策略
    # 每个策略需要什么指标，在这里复制一份到自己的策略里用

    # strategy_basic(positionEntity, inicatorEntity, windowDF, bar_num-1)  # 比如时间窗口60，最后一条数据下标永远是59
    strategy_t(positionEntity, inicatorEntity, windowDF, bar_num - 1, strategy_result, IEMultiLevel)


def strategy_basic(positionEntity, inicatorEntity, windowDF, DFLastRow):
    # 1、计算自己需要的指标
    windowDF = RMQIndicator.calMA(windowDF)

    # 2、执行策略
    if 0 == len(positionEntity.currentOrders):  # 空仓时买
        if inicatorEntity.tick_close < 0.99 * windowDF.iloc[DFLastRow]['MA_10']:  # 最新价格跟平均价格比较
            price = inicatorEntity.tick_close + 0.01  # 价格自己定，为防止买不进来，多挂点价格
            '''
            系数 = posiotionManage() # 写个仓位管理方法
            volume = self.money * 系数（0.几或 百分之几）
            '''
            volume = int(positionEntity.money / inicatorEntity.tick_close / 100) * 100
            # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
            RMQPosition.buy(positionEntity, inicatorEntity.tick_time, price, volume)  # 价格低于均价超过5%，买

    elif 0 != len(positionEntity.currentOrders):  # 如果不为0，说明买过，有仓位，那就可以卖，现在是全仓卖
        # 或者价格高于均价5%，可以卖——如果之前有仓位的话
        if inicatorEntity.tick_close > windowDF.iloc[DFLastRow]['MA_10'] * 1.01:
            # 把当前仓位的第一个卖掉
            key = list(positionEntity.currentOrders.keys())[0]
            # 加入T+1限制，判断当天买的，那就不能卖
            if inicatorEntity.tick_time.date() != positionEntity.currentOrders[key]['openDateTime'].date():
                price = inicatorEntity.tick_close + 0.01  # 价格自己定，为防止卖不出去，多挂点价格
                RMQPosition.sell(positionEntity, inicatorEntity.tick_time, key, price)


"""
"""
"""
论文方向：
没有一个算法总比其他算法好，
没有一个策略能适用所有情况。
解决办法就是，对现实情况进行概率分布假设，为不同分布采用不同算法。
99:1球的例子中，为什么改自己的猜测，因为我们更倾向概率大的，量化市场中，永远要选大概率。
对于策略，就是分析当前是哪种情况，采用不同策略。变化之后再分析，再变化策略。
——混合模型
数据准备
    按照各种策略先回测，把回测的数据，买入后还跌的剔除掉，买入后没涨的剔除掉，这样过滤一下，就拿到优质数据了。
    kaggle，证券宝等，用国内数据和数字币
混合模型、趋势预测 为主
辅助以资产组合优化、风控（仓位管理），情绪值直接调用交易所数据比如恐慌指数，
再加异常检测
这样目前的所有领域全占了

多看23年、24年的论文

清理github star
3月，啃完策略书，开始啃AI策略，记录学习过程顺便攒粉
    https://github.com/HanLi123/book/tree/master/%E4%BB%A3%E7%A0%81
    多用gpt做策略代码
    2.4节，最优控制，涉及第7章，基于最优控制进行回撤控制。gpt实现？
    代码果然写错了，用李涵的github代码再改改
精通pytorch，调通船舶
开始微调模型，先弄说英语那个
3月，开AI时，同时使用流行的模型工具，并包装成展品给亨，然后找商机

量化：1、展示赚钱的策略（短，先抛结果，再讲原理，obs）；2、然后策略的实盘信号推送
（短图文实盘）股吧，同花顺，雪球，币安：AI量化公众号，AI量化知识星球
1个B站号,1个知识星球
"""


"""
5，均值回归策略

多个级别买点同时出现
每个标的5k或10k，等周线新低，且来信号，买入，2%止损（能避免目前所有亏损 16%，一年最大亏3次，才6%，只要有日线为前提的多级别信号嵌套，亏损概率很小）。
或不按数值，趋势没形成就止损，判断依据就是当前级别结束时，看叉到底形成没有，如果没有，就发止损信号。
那其实就是判断，底背离买入后，是否真的红柱出现，出现那没问题
如果没有出现，反而绿柱继续扩大，说明是假信号，要止损。 这个再看吧
后面再出信号，再买，因为此时大级别信号还在失效期内
后面没有信号，不买，这就避免了被套
顶背离止盈（光伏这次） 或 最高位回撤2%止盈（农业那次）。
不硬抗，亏的少
占用资金周期短，资金量大。
保守的信号出现后大资金买入，第二天再次确认信号，没涨就直接清仓。这样虽然可能错过，但风险绝对低，只要赚一次就够了。一年总会有几次机会。
多等，少动，理智。
长线可能踏空，但我活的久，没有被套风险
盈利取决于反转或反弹高度

如果开通融券，就加个反向做空，先把代码加上
"""
def strategy_t(positionEntity, inicatorEntity, windowDF, DFLastRow, strategy_result, IEMultiLevel):
    # 1、计算自己需要的指标
    divergeDF = RMQAnalyse.calMACD_area(windowDF)  # df第0条是当前区域，第1条是过去的区域
    # 2、更新多级别指标对象
    if inicatorEntity.IE_timeLevel == 'd':
        # 目前只用到day级别，所以加这个判断和下面 kdj判断不等于d，都是为了减少计算量，提升系统速度
        windowDF = RMQIndicator.calKDJ(windowDF)  # 计算KDJ
        IEMultiLevel.updateInicatorEntityMultiLevel(inicatorEntity, windowDF, DFLastRow) # 更新多级别指标对象
    # 3、执行策略

    # print(divergeDF)
    # 底背离判断
    if divergeDF.iloc[2]['area'] < 0:
        # macd绿柱面积过去 > 现在  因为是负数，所以要更小
        if divergeDF.iloc[2]['area'] < divergeDF.iloc[0]['area'] and \
                divergeDF.iloc[2]['price'] > divergeDF.iloc[0]['price']:  # 过去最低价 > 现在最低价
            # KDJ判断
            if inicatorEntity.IE_timeLevel != 'd':
                windowDF = RMQIndicator.calKDJ(windowDF)  # 计算KDJ
            # 当前K上穿D，金叉
            # df最后一条数据就是最新的，又因为时间窗口固定，最后一条下标是DFLastRow
            if windowDF.iloc[DFLastRow]['K'] > windowDF.iloc[DFLastRow]['D'] and \
                    windowDF.iloc[DFLastRow - 1]['K'] < windowDF.iloc[DFLastRow - 1]['D']:
                # KDJ在超卖区
                if windowDF.iloc[DFLastRow]['K'] < 20 and windowDF.iloc[DFLastRow]['D'] < 20:
                    # 日线指标已更新，对比后再决定买卖
                    if IEMultiLevel.level_day_K is not None and IEMultiLevel.level_day_K < 20:  # 日线KDJ的K小于20再买
                        # 把tick时间转为字符串，下面要用
                        # '%Y-%m-%d %H:%M'  每分钟都提示太频繁，改为天
                        tick_time = inicatorEntity.tick_time.strftime('%Y-%m-%d')
                        # 满足条件说明还没锁
                        if tick_time != inicatorEntity.last_msg_time_1:
                            # 编辑信息
                            # print("当前可能底背离+KDJ金叉，买：", divergeDF.iloc[2]['time'], "~", divergeDF.iloc[0]['time'],
                            #       inicatorEntity.tick_time)
                            post_msg = inicatorEntity.IE_assetsName + "-" + inicatorEntity.IE_assetsCode + "-" + \
                                       inicatorEntity.IE_timeLevel + "：目前底背离+KDJ金叉，买。 价格：" + \
                                       str(round(inicatorEntity.tick_close, 3)) + " 时间：" + \
                                       inicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
                            print(post_msg)
                            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                            trade_point = [tick_time, round(inicatorEntity.tick_close, 3), "buy"]
                            positionEntity.trade_point_list.append(trade_point)
                            # 设置推送消息
                            strategy_result.editMsg(inicatorEntity, post_msg)

                            # 下单
                            if 0 == len(positionEntity.currentOrders):  # 空仓时买
                                price = inicatorEntity.tick_close + 0.01  # 价格自己定，为防止买不进来，多挂点价格
                                volume = int(positionEntity.money / inicatorEntity.tick_close / 100) * 100
                                # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
                                RMQPosition.buy(positionEntity, inicatorEntity.tick_time, price, volume)  # 价格低于均价超过5%，买

                            # 更新锁
                            inicatorEntity.last_msg_time_1 = tick_time

    # 顶背离判断
    elif divergeDF.iloc[2]['area'] > 0:
        if divergeDF.iloc[2]['area'] > divergeDF.iloc[0]['area'] and \
                divergeDF.iloc[2]['price'] < divergeDF.iloc[0]['price']:
            # KDJ判断
            if inicatorEntity.IE_timeLevel != 'd':
                windowDF = RMQIndicator.calKDJ(windowDF)  # 计算KDJ
            # 当前K下穿D，死叉
            if windowDF.iloc[DFLastRow]['K'] < windowDF.iloc[DFLastRow]['D'] and \
                    windowDF.iloc[DFLastRow - 1]['K'] > windowDF.iloc[DFLastRow - 1]['D']:
                # KDJ在超买区
                if windowDF.iloc[DFLastRow]['K'] > 80 and windowDF.iloc[DFLastRow]['D'] > 80:
                    # 日线指标已更新，对比后再决定买卖
                    if IEMultiLevel.level_day_K is not None and IEMultiLevel.level_day_K > 50:  # 日线KDJ的K大于50再卖
                        # 把tick时间转为字符串，下面要用
                        # '%Y-%m-%d %H:%M'  每分钟都提示太频繁，改为天
                        tick_time = inicatorEntity.tick_time.strftime('%Y-%m-%d')
                        # 满足条件说明还没锁
                        if tick_time != inicatorEntity.last_msg_time_2:
                            # 编辑信息
                            # print("当前可能顶背离+KDJ死叉，卖：", divergeDF.iloc[2]['time'], "~", divergeDF.iloc[0]['time'],
                            #       inicatorEntity.tick_time)
                            post_msg = inicatorEntity.IE_assetsName + "-" + inicatorEntity.IE_assetsCode + "-" + \
                                       inicatorEntity.IE_timeLevel + "：目前顶背离+KDJ死叉，卖。 价格：" + \
                                       str(round(inicatorEntity.tick_close, 3)) + " 时间：" + \
                                       inicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
                            print(post_msg)
                            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                            trade_point = [tick_time, round(inicatorEntity.tick_close, 3), "sell"]
                            positionEntity.trade_point_list.append(trade_point)
                            # 设置推送消息
                            strategy_result.editMsg(inicatorEntity, post_msg)

                            # 下单
                            if 0 != len(positionEntity.currentOrders):  # 如果不为0，说明买过，有仓位，那就可以卖，现在是全仓卖
                                key = list(positionEntity.currentOrders.keys())[0]  # 把当前仓位的第一个卖掉
                                # 加入T+1限制，判断当天买的，那就不能卖
                                if inicatorEntity.tick_time.date() != positionEntity.currentOrders[key][
                                    'openDateTime'].date():
                                    price = inicatorEntity.tick_close + 0.01  # 价格自己定，为防止卖不出去，多挂点价格
                                    RMQPosition.sell(positionEntity, inicatorEntity.tick_time, key, price)

                            # 更新锁
                            inicatorEntity.last_msg_time_2 = tick_time

    else:
        print("区域面积为0，小概率情况，忽略")


# 狂人五日线也属于双均线策略
"""
4，趋势策略
看看光伏，背离还是不如均线趋势，下跌趋势风险太大，抄底不如等反转确认，虽然会少赚几个点，但能少亏几十个点。
要不要加入均线辅助判断趋势
要不就只做趋势，不抄底，只找底部刚起来的。涨了很久趋势一旦反转就退出
"""
def double_moving_average_strategy(df, short_window=5, long_window=10):
    # 计算收盘价的移动平均线
    close_ma_short = df['close'].rolling(window=short_window).mean()
    close_ma_long = df['close'].rolling(window=long_window).mean()

    # 根据移动平均线交叉的情况进行买入和卖出
    try:
        for i in range(1, len(df)):
            if close_ma_short[i] > close_ma_long[i] and close_ma_short[i-1] <= close_ma_long[i-1]:
                print("买")
            elif close_ma_short[i] < close_ma_long[i] and close_ma_short[i-1] >= close_ma_long[i-1]:
                print("卖")
    except Exception as e:  # 有几天是没有5日均线的，所以用except处理异常
        pass
    # 返回每日收益率和累计收益率
    return ""