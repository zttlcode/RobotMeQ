import numpy as np
import RMQData.Position as RMQPosition
import RMQData.Indicator as RMQIndicator
import RMQStrategy.Strategy_fuzzy as RMQSFuzzy
from RMQTool import Message


class StrategyResultEntity:
    def __init__(self):
        self.live = True
        self.msg_level_5 = "无"
        self.msg_level_15 = "无"
        self.msg_level_30 = "无"
        self.msg_level_60 = "无"
        self.msg_level_day = "无"

    # 组合多级别策略结果，拼接发送交易建议
    def send_msg(self, strategyName, indicatorEntity, IEMultiLevel, msg):
        # 编辑标题
        title = strategyName
        # 组织消息
        mail_list_qq = "mail_list_qq_d"
        # 设置邮箱地址
        if indicatorEntity.IE_timeLevel == "5":
            mail_list_qq = "mail_list_qq_5"
        elif indicatorEntity.IE_timeLevel == "15":
            mail_list_qq = "mail_list_qq_15"
        elif indicatorEntity.IE_timeLevel == "30":
            mail_list_qq = "mail_list_qq_30"
        elif indicatorEntity.IE_timeLevel == "60":
            mail_list_qq = "mail_list_qq_60"
        elif indicatorEntity.IE_timeLevel == "d":
            mail_list_qq = "mail_list_qq_d"

        # 设置消息
        if msg is not None:  # 是保守策略 或 ride mood策略
            post_msg = (indicatorEntity.IE_assetsName
                        + "-"
                        + indicatorEntity.IE_assetsCode
                        + "-"
                        + indicatorEntity.IE_timeLevel
                        + "：" + msg + "："
                        + str(round(indicatorEntity.tick_close, 3))
                        + " 时间："
                        + indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'))

            if indicatorEntity.IE_timeLevel == "5":
                self.msg_level_5 = post_msg
            elif indicatorEntity.IE_timeLevel == "15":
                self.msg_level_15 = post_msg
            elif indicatorEntity.IE_timeLevel == "30":
                self.msg_level_30 = post_msg
            elif indicatorEntity.IE_timeLevel == "60":
                self.msg_level_60 = post_msg
            elif indicatorEntity.IE_timeLevel == "d":
                self.msg_level_day = post_msg

        else:  # 是多级别激进策略
            self.msg_level_5 = "" if IEMultiLevel.level_5_diverge is None else IEMultiLevel.level_5_diverge
            self.msg_level_15 = "" if IEMultiLevel.level_15_diverge is None else IEMultiLevel.level_15_diverge
            self.msg_level_30 = "" if IEMultiLevel.level_30_diverge is None else IEMultiLevel.level_30_diverge
            self.msg_level_60 = "" if IEMultiLevel.level_60_diverge is None else IEMultiLevel.level_60_diverge
            self.msg_level_day = "" if IEMultiLevel.level_day_diverge is None else IEMultiLevel.level_day_diverge

        mail_msg = Message.build_msg_HTML(title, self)
        if self.live:
            # 需要异步发送，此函数还没写，暂时先同步发送
            res = Message.QQmail(mail_msg, mail_list_qq)
            if res:
                print('发送成功')
            else:
                print('发送失败')

    # 综合判断多个策略结果
    def synthetic_judge(self):
        pass


def strategy(asset, strategy_result, IEMultiLevel):
    barEntity = asset.barEntity
    indicatorEntity = asset.indicatorEntity
    positionEntity = asset.positionEntity

    # 1、每次tick重新计算指标数据
    # EMA DIF DEA MACD MA_5 MA_10 MA_60 K D J RSI6 RSI12 RSI24
    # start = datetime.now()

    # barEntity.bar_DataFrame = RMQIndicator.calMA(barEntity.bar_DataFrame)
    # barEntity.bar_DataFrame = RMQIndicator.calMACD(barEntity.bar_DataFrame)
    # barEntity.bar_DataFrame = RMQIndicator.calKDJ(barEntity.bar_DataFrame)
    # barEntity.bar_DataFrame = RMQIndicator.calRSI(barEntity.bar_DataFrame)
    # 耗时样例：end 0:00:00.109344 1014  耗时随规模线性增长，时间复杂度为O（n）

    # 指标计算为了省时间，设置个固定数据量的时间窗口
    length = len(barEntity.bar_DataFrame)
    window = length - barEntity.bar_num  # 起始下标比如是0~60，100~160等，bar_num是60，iloc含头不含尾。现在bar_num是250
    windowDF = barEntity.bar_DataFrame.iloc[window:length].copy()  # copy不改变原对象，不加copy会有改变临时对象的警告
    windowDF = windowDF.reset_index(drop=True)  # 重置索引，这样df索引总是0~59

    """
    2024 04 27 之前一直把实时价格当作windowDF最新一条计算指标，以为这就是实盘的核心，实践证明是错的
    当前这个bar，只有到收盘那一刻，才能确认指标，盘中就算到了预期点位但收盘没到，那就不算反转，虚晃一枪
    因此计算指标要去掉windowDF最新一条(索引最大的)。实时价格只用来对比是否到了止损位
    """
    windowDF_calIndic = windowDF.iloc[:-1].copy()  # copy不改变原对象，不加copy会有改变临时对象的警告
    windowDF_calIndic = windowDF_calIndic.reset_index(drop=True)  # 重置索引，这样df索引是0~58

    # windowDF = RMQIndicator.calMA(windowDF)
    # windowDF = RMQIndicator.calMACD(windowDF)
    # windowDF = RMQIndicator.calKDJ(windowDF)
    # windowDF = RMQIndicator.calRSI(windowDF)
    # 耗时样例：end 0:00:00.015624 1449  耗时不随规模增长，一直保持在15毫秒，时间复杂度为O（1）
    # print("end", datetime.now()-start, len(barEntity.bar_DataFrame), len(windowDF))
    # 每个tick都要重新算，特别耗时  时间窗口对性能提升很大 用windowDF，不要用上面的 indicatorEntity.bar_DataFrame
    # -----------------------------------------------------------------------------------------------

    # 保守派背离策略  每个级别各玩各的，互不打扰  2024-4月~2026-4月 运行在阿里云服务器
    strategy_tea_conservative(positionEntity,
                              indicatorEntity,
                              windowDF_calIndic,
                              barEntity.bar_num - 2,  # 比如时间窗口60，最后一条数据下标永远是59
                              strategy_result,
                              IEMultiLevel)

    # 激进派背离策略，
    # strategy_tea_radical(positionEntity,
    #                      indicatorEntity,
    #                      windowDF_calIndic,
    #                      barEntity.bar_num - 2,  # 比如时间窗口60，最后一条数据下标永远是59
    #                      strategy_result,
    #                      IEMultiLevel)

    # ride-mood策略，增强版趋势跟随策略，反指率高达90%，经常小亏，偶尔大赚
    # strategy_fuzzy(positionEntity,
    #                indicatorEntity,
    #                windowDF_calIndic,
    #                barEntity.bar_num - 1,  # 减了个实时价格，250变249，所以这里长度也跟着变成249
    #                strategy_result)


def strategy_tea_conservative(positionEntity,
                              indicatorEntity,
                              windowDF_calIndic,
                              DFLastRow,
                              strategy_result,
                              IEMultiLevel):
    if 0 != len(positionEntity.currentOrders):  # 满仓，判断止损
        RMQPosition.stopLoss(positionEntity, indicatorEntity, strategy_result)

    """
    这个逻辑跟bar_generator那个类似，但这个粒度细，没有特殊注意点,
    如果每个级别各按自己的时间级别来锁，那就又要写那个很长的if条件，因为a股有半点有整点，上午下午不一样，麻烦
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            # 1、计算自己需要的指标
            divergeDF = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
            # 2、更新多级别指标对象
            if indicatorEntity.IE_timeLevel == 'd':
                # 目前只用到day级别，所以加这个判断和下面 kdj判断不等于d，都是为了减少计算量，提升系统速度
                windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                IEMultiLevel.updateDayK(windowDF_calIndic, DFLastRow)  # 更新多级别指标对象
            # 3、执行策略

            # 空仓
            if 0 == len(positionEntity.currentOrders):
                # 底背离判断
                if divergeDF.iloc[2]['area'] < 0:
                    # macd绿柱面积过去 > 现在  因为是负数，所以要更小
                    if (divergeDF.iloc[2]['area'] < divergeDF.iloc[0]['area']
                            and divergeDF.iloc[2]['price'] > divergeDF.iloc[0]['price']):  # 过去最低价 > 现在最低价
                        # KDJ判断
                        if indicatorEntity.IE_timeLevel != 'd':
                            windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                        # 当前K上穿D，金叉
                        # df最后一条数据就是最新的，又因为时间窗口固定，最后一条下标是DFLastRow
                        if (windowDF_calIndic.iloc[DFLastRow]['K']
                                > windowDF_calIndic.iloc[DFLastRow]['D']
                                and
                                windowDF_calIndic.iloc[DFLastRow - 1]['K']
                                < windowDF_calIndic.iloc[DFLastRow - 1]['D']):
                            # KDJ在超卖区
                            if (windowDF_calIndic.iloc[DFLastRow]['K'] < 20
                                    and
                                    windowDF_calIndic.iloc[DFLastRow]['D'] < 20):
                                # 日线指标已更新，对比后再决定买卖
                                if (IEMultiLevel.level_day_K is not None
                                        and
                                        IEMultiLevel.level_day_K < 20):  # 日线KDJ的K小于20再买
                                    # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                                    trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H'),
                                                   round(indicatorEntity.tick_close, 3),
                                                   "buy"]
                                    positionEntity.trade_point_list.append(trade_point)
                                    # 推送消息
                                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                                             + "-"
                                                             + indicatorEntity.IE_assetsCode,
                                                             indicatorEntity,
                                                             None,
                                                             "目前底背离+KDJ金叉")

                                    # price = indicatorEntity.tick_close + 0.01  # 价格自己定，为防止买不进来，多挂点价格
                                    price = indicatorEntity.tick_close
                                    volume = int(positionEntity.money / price / 100) * 100
                                    # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
                                    RMQPosition.buy(positionEntity,
                                                    indicatorEntity,
                                                    price,
                                                    volume)  # 价格低于均价超过5%，买
            # 满仓
            if 0 != len(positionEntity.currentOrders):
                # 顶背离判断
                if divergeDF.iloc[2]['area'] > 0:
                    if (divergeDF.iloc[2]['area'] > divergeDF.iloc[0]['area']
                            and
                            divergeDF.iloc[2]['price'] < divergeDF.iloc[0]['price']):
                        # KDJ判断
                        if indicatorEntity.IE_timeLevel != 'd':
                            windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                        # 当前K下穿D，死叉
                        if (windowDF_calIndic.iloc[DFLastRow]['K']
                                < windowDF_calIndic.iloc[DFLastRow]['D']
                                and
                                windowDF_calIndic.iloc[DFLastRow - 1]['K']
                                > windowDF_calIndic.iloc[DFLastRow - 1]['D']):
                            # KDJ在超买区
                            if (windowDF_calIndic.iloc[DFLastRow]['K'] > 80
                                    and
                                    windowDF_calIndic.iloc[DFLastRow]['D'] > 80):
                                # 日线指标已更新，对比后再决定买卖
                                if (IEMultiLevel.level_day_K is not None
                                        and IEMultiLevel.level_day_K > 50):  # 日线KDJ的K大于50再卖
                                    # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                                    trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H'),
                                                   round(indicatorEntity.tick_close, 3),
                                                   "sell"]
                                    positionEntity.trade_point_list.append(trade_point)
                                    # 设置推送消息
                                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                                             + "-"
                                                             + indicatorEntity.IE_assetsCode,
                                                             indicatorEntity,
                                                             None,
                                                             "目前顶背离+KDJ死叉")

                                    # # 加入T+1限制，判断当天买的，那就不能卖   2024 04 27 A股几乎不可能日内指标反转
                                    # if (indicatorEntity.tick_time.date()
                                    #         != positionEntity.currentOrders[key]['openDateTime'].date()):
                                    #     price = indicatorEntity.tick_close + 0.01  # 价格自己定，为防止卖不出去，多挂点价格
                                    #     RMQPosition.sell(positionEntity,
                                    #                      indicatorEntity.tick_time.strftime('%Y-%m-%d'),
                                    #                      key,
                                    #                      price)
                                    RMQPosition.sell(positionEntity, indicatorEntity)


def strategy_tea_radical(positionEntity,
                         indicatorEntity,
                         windowDF_calIndic,
                         DFLastRow,
                         strategy_result,
                         IEMultiLevel):
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            # 1、计算自己需要的指标
            divergeDF = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
            # 2、更新多级别指标对象
            if indicatorEntity.IE_timeLevel == 'd':
                # 目前只用到day级别，所以加这个判断和下面 kdj判断不等于d，都是为了减少计算量，提升系统速度
                windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                IEMultiLevel.updateDayK(windowDF_calIndic, DFLastRow)  # 更新多级别指标对象
            # 3、执行策略
            if divergeDF.iloc[2]['area'] < 0:  # 底背离判断
                # macd绿柱面积过去 > 现在  因为是负数，所以要更小
                if (divergeDF.iloc[2]['area']
                        < divergeDF.iloc[0]['area']
                        and
                        divergeDF.iloc[2]['price']
                        > divergeDF.iloc[0]['price']):  # 过去最低价 > 现在最低价
                    # KDJ判断
                    if indicatorEntity.IE_timeLevel != 'd':
                        windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                    # 当前K上穿D，金叉
                    # df最后一条数据就是最新的，又因为时间窗口固定，最后一条下标是DFLastRow
                    if (windowDF_calIndic.iloc[DFLastRow]['K']
                            > windowDF_calIndic.iloc[DFLastRow]['D']
                            and
                            windowDF_calIndic.iloc[DFLastRow - 1]['K']
                            < windowDF_calIndic.iloc[DFLastRow - 1]['D']):
                        # KDJ在超卖区
                        if (windowDF_calIndic.iloc[DFLastRow]['K'] < 35  # 20
                                and windowDF_calIndic.iloc[DFLastRow]['D'] < 35):  # 20
                            tempK = 35  # 20
                            if indicatorEntity.IE_assetsCode == '510300':
                                tempK = 50
                            # 日线指标已更新，对比后再决定买卖
                            if (IEMultiLevel.level_day_K is not None
                                    and IEMultiLevel.level_day_K < tempK):  # 日线KDJ的K小于20再买
                                # 更新指标信号：底背离 第一个区域面积
                                isUpdated = indicatorEntity.updateSignal(0,
                                                                         round(divergeDF.iloc[2]['area'], 3),
                                                                         round(indicatorEntity.tick_close, 3))
                                if isUpdated:
                                    # 将信号信息更新到多级别对象
                                    IEMultiLevel.updateDiverge(indicatorEntity)

                                    # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                                    trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                   round(indicatorEntity.tick_close, 3),
                                                   "buy"]
                                    positionEntity.trade_point_list.append(trade_point)
                                    # 推送消息
                                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                                             + "-"
                                                             + indicatorEntity.IE_assetsCode,
                                                             indicatorEntity,
                                                             IEMultiLevel,
                                                             None)
                                    # 买 RMQPosition.buy(positionEntity, indicatorEntity, indicatorEntity.tick_close,
                                    # int(positionEntity.money / indicatorEntity.tick_close / 100) * 100)

            if divergeDF.iloc[2]['area'] > 0:  # 顶背离判断
                if (divergeDF.iloc[2]['area']
                        > divergeDF.iloc[0]['area']
                        and
                        divergeDF.iloc[2]['price']
                        < divergeDF.iloc[0]['price']):
                    # KDJ判断
                    if indicatorEntity.IE_timeLevel != 'd':
                        windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                    # 当前K下穿D，死叉
                    if (windowDF_calIndic.iloc[DFLastRow]['K']
                            < windowDF_calIndic.iloc[DFLastRow]['D']
                            and
                            windowDF_calIndic.iloc[DFLastRow - 1]['K']
                            > windowDF_calIndic.iloc[DFLastRow - 1]['D']):
                        # KDJ在超买区
                        if (windowDF_calIndic.iloc[DFLastRow]['K'] > 80
                                and windowDF_calIndic.iloc[DFLastRow]['D'] > 80):
                            # 日线指标已更新，对比后再决定买卖
                            if (IEMultiLevel.level_day_K is not None
                                    and IEMultiLevel.level_day_K > 50):  # 日线KDJ的K大于50再卖

                                # 更新指标信号：顶背离 第一个区域面积
                                isUpdated = indicatorEntity.updateSignal(1,
                                                                         round(divergeDF.iloc[2]['area'], 3),
                                                                         round(indicatorEntity.tick_close, 3))
                                if isUpdated:
                                    # 将信号信息更新到多级别对象
                                    IEMultiLevel.updateDiverge(indicatorEntity)

                                    # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                                    trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                   round(indicatorEntity.tick_close, 3),
                                                   "sell"]
                                    positionEntity.trade_point_list.append(trade_point)
                                    # 设置推送消息
                                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                                             + "-"
                                                             + indicatorEntity.IE_assetsCode,
                                                             indicatorEntity,
                                                             IEMultiLevel,
                                                             None)
                                    # 卖
                                    # RMQPosition.sell(positionEntity, indicatorEntity)


def strategy_fuzzy(positionEntity,
                   indicatorEntity,
                   windowDF_calIndic,
                   bar_num,
                   strategy_result):
    if 0 != len(positionEntity.currentOrders):  # 满仓，判断止损
        RMQPosition.stopLoss(positionEntity, indicatorEntity, strategy_result)

    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            n1, n2, aa = RMQSFuzzy.strategy_fuzzy(windowDF_calIndic, bar_num)
            # bar_num为了算过去的指标，250-1，所以n2是249,最后一位下标248，没值，243~247有值，
            mood = aa[1, 0, n2 - 6:n2 - 1] - aa[0, 0, n2 - 6:n2 - 1]  # a7-a6的值，正：可以买
            avmood = np.mean(mood)

            # 空仓，且大买家占优则买
            if 0 == len(positionEntity.currentOrders) and avmood > 0:  # 空仓时买
                # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H'),
                               round(indicatorEntity.tick_close, 3),
                               "buy"]
                positionEntity.trade_point_list.append(trade_point)
                # 推送消息
                strategy_result.send_msg(indicatorEntity.IE_assetsName
                                         + "-"
                                         + indicatorEntity.IE_assetsCode,
                                         indicatorEntity,
                                         None,
                                         "buy" + str(round(avmood, 3)))

                volume = int(positionEntity.money / indicatorEntity.tick_close / 100) * 100
                # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
                RMQPosition.buy(positionEntity, indicatorEntity, indicatorEntity.tick_close, volume)
            # 满仓
            if 0 != len(positionEntity.currentOrders) and avmood < 0:
                # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H'),
                               round(indicatorEntity.tick_close, 3),
                               "sell"]
                positionEntity.trade_point_list.append(trade_point)
                # 设置推送消息
                strategy_result.send_msg(indicatorEntity.IE_assetsName
                                         + "-"
                                         + indicatorEntity.IE_assetsCode,
                                         indicatorEntity,
                                         None,
                                         "sell" + str(round(avmood, 3)))
                # 卖
                RMQPosition.sell(positionEntity, indicatorEntity)
