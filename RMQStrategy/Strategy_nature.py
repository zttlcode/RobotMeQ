import numpy as np
import RMQData.Position as RMQPosition
import RMQData.Indicator as RMQIndicator
import RMQStrategy.Strategy_fuzzy as RMQSFuzzy


def strategy_tea_radical(positionEntity,
                         indicatorEntity,
                         windowDF_calIndic,
                         DFLastRow,
                         strategy_result,
                         IEMultiLevel):
    # 1、计算自己需要的指标
    divergeDF, windowDF_calIndic = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
    try:
        # 3、执行策略
        if divergeDF.iloc[2]['area'] < 0:  # 底背离判断
            # macd绿柱面积过去 > 现在  因为是负数，所以要更小
            """
           2025 01 22
           看图发现，有些买点明显MACD变小，跌势加剧，但所处区域面积小，仍可算作底背离，导致买入，因此加校验
           可以在divergeDF中加，但怕改代码导致其他问题
           实盘增加计算macd柱子变化
            """
            if (divergeDF.iloc[2]['area']
                    < divergeDF.iloc[0]['area']
                    and
                    divergeDF.iloc[2]['price']
                    > divergeDF.iloc[0]['price']
                    and
                    windowDF_calIndic.iloc[DFLastRow]['MACD']
                    >= windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):  # 过去最低价 > 现在最低价
                # KDJ判断
                windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                # 当前K上穿D，金叉
                # df最后一条数据就是最新的，又因为时间窗口固定，最后一条下标是DFLastRow
                if (windowDF_calIndic.iloc[DFLastRow]['K']
                        > windowDF_calIndic.iloc[DFLastRow]['D']
                        and
                        windowDF_calIndic.iloc[DFLastRow - 1]['K']
                        < windowDF_calIndic.iloc[DFLastRow - 1]['D']):
                    # KDJ在超卖区
                    tempK = 35  # 20
                    if (windowDF_calIndic.iloc[DFLastRow]['K'] < tempK
                            and windowDF_calIndic.iloc[DFLastRow]['D'] < tempK):
                        # 更新指标信号：底背离 第一个区域面积
                        isUpdated = indicatorEntity.updateSignal(0,
                                                                 round(divergeDF.iloc[2]['area'], 3),
                                                                 round(indicatorEntity.tick_close, 3))
                        if isUpdated:
                            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                            trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                           round(indicatorEntity.tick_close, 3),
                                           "buy"]
                            positionEntity.trade_point_list.append(trade_point)

        if divergeDF.iloc[2]['area'] > 0:  # 顶背离判断
            if (divergeDF.iloc[2]['area']
                    > divergeDF.iloc[0]['area']
                    and
                    divergeDF.iloc[2]['price']
                    < divergeDF.iloc[0]['price']
                    and
                    windowDF_calIndic.iloc[DFLastRow]['MACD']
                    <= windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):
                # KDJ判断
                # if indicatorEntity.IE_timeLevel != 'd':
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
                        # 更新指标信号：顶背离 第一个区域面积
                        isUpdated = indicatorEntity.updateSignal(1,
                                                                 round(divergeDF.iloc[2]['area'], 3),
                                                                 round(indicatorEntity.tick_close, 3))
                        if isUpdated:
                            # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
                            trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                           round(indicatorEntity.tick_close, 3),
                                           "sell"]
                            positionEntity.trade_point_list.append(trade_point)
    except Exception as e:
        print("Error happens ", indicatorEntity.IE_assetsCode, " ", indicatorEntity.IE_timeLevel, " ", e)


def strategy_fuzzy(positionEntity,
                   indicatorEntity,
                   windowDF_calIndic,
                   bar_num,
                   strategy_result):
    # if 0 != len(positionEntity.currentOrders):  # 满仓，判断止损
    #     RMQPosition.stopLoss(positionEntity, indicatorEntity, strategy_result)

    # current_min = int(indicatorEntity.tick_time.strftime('%M'))
    # if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
    #     if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
    #         indicatorEntity.last_cal_time = current_min  # 更新锁

    n1, n2, aa = RMQSFuzzy.fuzzy(windowDF_calIndic, bar_num)
    # bar_num为了算过去的指标，250-1，所以n2是249,最后一位下标248，没值，243~247有值，
    mood = aa[1, 0, n2 - 6:n2 - 1] - aa[0, 0, n2 - 6:n2 - 1]  # a7-a6的值，正：可以买
    avmood = np.mean(mood)

    # 空仓，且大买家占优则买
    if 0 == len(positionEntity.currentOrders) and avmood > 0:  # 空仓时买
        # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "buy"]

        positionEntity.trade_point_list.append(trade_point)
        # # 推送消息
        # strategy_result.send_msg(indicatorEntity.IE_assetsName
        #                          + "-"
        #                          + indicatorEntity.IE_assetsCode,
        #                          indicatorEntity,
        #                          None,
        #                          "buy" + str(round(avmood, 3)))

        volume = int(positionEntity.money / indicatorEntity.tick_close / 100) * 100
        # 全仓买,1万本金除以股价，算出能买多少股，# 再除以100算出能买多少手，再乘100算出要买多少股
        RMQPosition.buy(positionEntity, indicatorEntity, indicatorEntity.tick_close, volume)
    # 满仓
    if 0 != len(positionEntity.currentOrders) and avmood < 0:
        # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "sell"]

        positionEntity.trade_point_list.append(trade_point)
        # # 设置推送消息
        # strategy_result.send_msg(indicatorEntity.IE_assetsName
        #                          + "-"
        #                          + indicatorEntity.IE_assetsCode,
        #                          indicatorEntity,
        #                          None,
        #                          "sell" + str(round(avmood, 3)))
        # 卖
        RMQPosition.sell(positionEntity, indicatorEntity)
