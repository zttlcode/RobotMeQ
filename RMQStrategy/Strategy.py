import RMQStrategy.Position as RMQPosition
import RMQStrategy.Indicator as RMQIndicator
import RMQStrategy.Indicator_analyse as RMQAnalyse
import RMQStrategy.Strategy_fuzzy as RMQSFuzzy
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
            title = "均值回归多级别策略"
            # 组织消息
            mail_msg = Message.build_msg_HTML(title, self)
            # 需要异步发送，此函数还没写，暂时先同步发送
            res = Message.QQmail(mail_msg)
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

    # strategy_t(positionEntity, inicatorEntity, windowDF, bar_num - 1, strategy_result, IEMultiLevel)  #
    # 比如时间窗口60，最后一条数据下标永远是59
    strategy_f(positionEntity, inicatorEntity, windowDF, bar_num, strategy_result, IEMultiLevel)


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
                                       inicatorEntity.IE_timeLevel + "：目前底背离+KDJ金叉：" + \
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
                                       inicatorEntity.IE_timeLevel + "：目前顶背离+KDJ死叉：" + \
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


def strategy_f(positionEntity, inicatorEntity, windowDF, bar_num, strategy_result, IEMultiLevel):
    n1, n2, aa = RMQSFuzzy.strategy_fuzzy(positionEntity, inicatorEntity, windowDF, bar_num, strategy_result, IEMultiLevel)
    mood_prv = aa[1, 0, bar_num-3] - aa[0, 0, bar_num-3]
    mood = aa[1, 0, bar_num-2] - aa[0, 0, bar_num-2]
    # 空仓，且大买家占有则下单
    if 0 == len(positionEntity.currentOrders) and mood_prv < 0 and mood > 0:  # 空仓时买
        # '%Y-%m-%d %H:%M'  每分钟都提示太频繁，改为分钟
        tick_time = inicatorEntity.tick_time.strftime('%Y-%m-%d %H')
        # 满足条件说明还没锁
        if tick_time != inicatorEntity.last_msg_time_1:
            # 编辑信息
            post_msg = inicatorEntity.IE_assetsName + "-" + inicatorEntity.IE_assetsCode + "-" + \
                       inicatorEntity.IE_timeLevel + "：买：" + str(round(mood, 3))+","+ \
                       str(round(inicatorEntity.tick_close, 3)) + " 时间：" + \
                       inicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
            print(post_msg)
            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
            trade_point = [tick_time, round(inicatorEntity.tick_close, 3), "buy"]
            positionEntity.trade_point_list.append(trade_point)
            # 设置推送消息
            # strategy_result.editMsg(inicatorEntity, post_msg)

            # 更新锁
            inicatorEntity.last_msg_time_1 = tick_time

            price = inicatorEntity.tick_close
            volume = int(positionEntity.money / inicatorEntity.tick_close / 100) * 100
            # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
            RMQPosition.buy(positionEntity, inicatorEntity.tick_time, price, volume)

    # 下单
    if 0 != len(positionEntity.currentOrders) and mood_prv>0 and mood < 0:  # 如果不为0，说明买过，有仓位，那就可以卖，现在是全仓卖
        key = list(positionEntity.currentOrders.keys())[0]  # 把当前仓位的第一个卖掉
        # '%Y-%m-%d %H:%M'  每分钟都提示太频繁，改为分钟
        tick_time = inicatorEntity.tick_time.strftime('%Y-%m-%d %H')
        # 满足条件说明还没锁
        if tick_time != inicatorEntity.last_msg_time_2:
            # 编辑信息
            # print("当前可能顶背离+KDJ死叉，卖：", divergeDF.iloc[2]['time'], "~", divergeDF.iloc[0]['time'],
            #       inicatorEntity.tick_time)
            post_msg = inicatorEntity.IE_assetsName + "-" + inicatorEntity.IE_assetsCode + "-" + \
                       inicatorEntity.IE_timeLevel + "：卖：" + str(round(mood, 3))+","+ \
                       str(round(inicatorEntity.tick_close, 3)) + " 时间：" + \
                       inicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
            print(post_msg)
            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
            trade_point = [tick_time, round(inicatorEntity.tick_close, 3), "sell"]
            positionEntity.trade_point_list.append(trade_point)
            # 设置推送消息
            # strategy_result.editMsg(inicatorEntity, post_msg)

            # 更新锁
            inicatorEntity.last_msg_time_2 = tick_time

            price = inicatorEntity.tick_close
            RMQPosition.sell(positionEntity, inicatorEntity.tick_time, key, price)

