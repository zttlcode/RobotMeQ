import RMQData.Position as RMQPosition
import RMQData.Indicator as RMQIndicator
import RMQTool.Run_live_model as RMQRun_live_model


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
            divergeDF, windowDF_calIndic = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
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
                            and divergeDF.iloc[2]['price'] > divergeDF.iloc[0]['price']
                            and windowDF_calIndic.iloc[DFLastRow]['MACD'] >=
                            windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):  # 过去最低价 > 现在最低价
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
                            and divergeDF.iloc[2]['price'] < divergeDF.iloc[0]['price']
                            and windowDF_calIndic.iloc[DFLastRow]['MACD'] <=
                            windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):
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
            divergeDF, windowDF_calIndic = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
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
                        > divergeDF.iloc[0]['price']
                        and windowDF_calIndic.iloc[DFLastRow]['MACD'] >=
                        windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):  # 过去最低价 > 现在最低价
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
                                    # # 推送消息
                                    # strategy_result.send_msg(indicatorEntity.IE_assetsName
                                    #                          + "-"
                                    #                          + indicatorEntity.IE_assetsCode,
                                    #                          indicatorEntity,
                                    #                          IEMultiLevel,
                                    #                          None)
                                    RMQRun_live_model.run_live_call_model(indicatorEntity,
                                                                          "buy")  # 2025 03 06 实盘调模型，不发消息
                                    # 买 RMQPosition.buy(positionEntity, indicatorEntity, indicatorEntity.tick_close,
                                    # int(positionEntity.money / indicatorEntity.tick_close / 100) * 100)

            if divergeDF.iloc[2]['area'] > 0:  # 顶背离判断
                if (divergeDF.iloc[2]['area']
                        > divergeDF.iloc[0]['area']
                        and divergeDF.iloc[2]['price']
                        < divergeDF.iloc[0]['price']
                        and windowDF_calIndic.iloc[DFLastRow]['MACD']
                        <= windowDF_calIndic.iloc[DFLastRow - 1]['MACD']):
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
                                    # # 设置推送消息
                                    # strategy_result.send_msg(indicatorEntity.IE_assetsName
                                    #                          + "-"
                                    #                          + indicatorEntity.IE_assetsCode,
                                    #                          indicatorEntity,
                                    #                          IEMultiLevel,
                                    #                          None)
                                    RMQRun_live_model.run_live_call_model(indicatorEntity,
                                                                          "sell")  # 2025 03 06 实盘调模型，不发消息
                                    # 卖
                                    # RMQPosition.sell(positionEntity, indicatorEntity)
