import numpy as np
import RMQData.Position as RMQPosition
import RMQData.Indicator as RMQIndicator
import RMQStrategy.Strategy_fuzzy as RMQSFuzzy

"""
    1、趋势 fuzzy，利用模糊理论，基于均线，判断当前是买方强势还是卖方强势，跟随强者操作。
    2、趋势 MACD（带signal）+ RSI + EMA60 + 收盘价：适合趋势市和中长期趋势，适合中长线交易者。
    3、趋势 MACD + KDJ + 收盘价：适合震荡市和短期趋势，适合短线交易者。
    4、震荡 boll + RSI + 收盘价：震荡策略。
    5、震荡 KDJ：震荡策略。
    6、突破 震荡转趋势 boll + ATR + 成交量 + 收盘价。
    7、反转 趋势转趋势 MACD（带signal）+ obv + ema10 + ema60。
"""


def strategy_tea_radical(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result, IEMultiLevel):
    # 1、计算自己需要的指标
    divergeDF, windowDF_calIndic = RMQIndicator.calMACD_area(windowDF_calIndic)  # df第0条是当前区域，第1条是过去的区域
    try:
        # 3、执行策略
        if divergeDF.iloc[2]['area'] < 0:  # 底背离判断
            # macd绿柱面积过去 > 现在  因为是负数，所以要更小
            """
           2025 01 22
           看图发现，有些买点明显MACD变小，跌势加剧，但所处区域面积小，仍可算作底背离，导致买入，因此加校验计算macd柱子变化
           windowDF_calIndic.iloc[DFLastRow]['MACD']
                    >= windowDF_calIndic.iloc[DFLastRow - 1]['MACD']
            """
            if (divergeDF.iloc[2]['area']
                    < divergeDF.iloc[0]['area']
                    and
                    divergeDF.iloc[2]['price']
                    > divergeDF.iloc[0]['price']
                    and
                    windowDF_calIndic.iloc[-1]['MACD']
                    >= windowDF_calIndic.iloc[-2]['MACD']):  # 过去最低价 > 现在最低价
                # KDJ判断
                windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                # 当前K上穿D，金叉
                # df最后一条数据就是最新的，又因为时间窗口固定，最后一条下标是DFLastRow
                if (windowDF_calIndic.iloc[-1]['K']
                        > windowDF_calIndic.iloc[-1]['D']
                        and
                        windowDF_calIndic.iloc[-2]['K']
                        < windowDF_calIndic.iloc[-2]['D']):
                    # KDJ在超卖区
                    if (windowDF_calIndic.iloc[-1]['K'] < 35  # 20
                            and windowDF_calIndic.iloc[-1]['D'] < 35):  # 20
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
                    windowDF_calIndic.iloc[-1]['MACD']
                    <= windowDF_calIndic.iloc[-2]['MACD']):
                # KDJ判断
                # if indicatorEntity.IE_timeLevel != 'd':
                windowDF_calIndic = RMQIndicator.calKDJ(windowDF_calIndic)  # 计算KDJ
                # 当前K下穿D，死叉
                if (windowDF_calIndic.iloc[-1]['K']
                        < windowDF_calIndic.iloc[-1]['D']
                        and
                        windowDF_calIndic.iloc[-2]['K']
                        > windowDF_calIndic.iloc[-2]['D']):
                    # KDJ在超买区
                    if (windowDF_calIndic.iloc[-1]['K'] > 80
                            and windowDF_calIndic.iloc[-1]['D'] > 80):
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


def strategy_fuzzy(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    # if 0 != len(positionEntity.currentOrders):  # 满仓，判断止损
    #     RMQPosition.stopLoss(positionEntity, indicatorEntity, strategy_result)

    # current_min = int(indicatorEntity.tick_time.strftime('%M'))
    # if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
    #     if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
    #         indicatorEntity.last_cal_time = current_min  # 更新锁

    n1, n2, aa = RMQSFuzzy.fuzzy(windowDF_calIndic)
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


def strategy_c4_trend(positionEntity, indicatorEntity, df):
    """趋势跟踪策略
    指标组合：均线（MA/EMA）+ MACD + ADX
    交易策略：顺势交易，逢低买入（上升趋势），逢高做空（下降趋势）
    """
    # 获取最新数据点
    last_row = df.iloc[-1]

    # 趋势方向判断
    trend_direction = 'up' if last_row['close'] > last_row['ema60'] else 'down'

    # 交易信号
    if trend_direction == 'up':
        # 多头策略
        if (last_row['macd'] > last_row['signal']) and (last_row['rsi'] < 30):
            # print(f"[趋势买入] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            # 这种情况不足1%，只能靠趋势平仓去买  感觉又成回归交易了
            pass
        elif (last_row['macd'] < last_row['signal']) and (last_row['rsi'] > 70):
            # print(f"[趋势卖出] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                           round(indicatorEntity.tick_close, 3),
                           "sell"]
            positionEntity.trade_point_list.append(trade_point)
    else:
        # 空头策略
        if (last_row['macd'] < last_row['signal']) and (last_row['rsi'] > 70):
            # print(f"[趋势做空] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            # 跟趋势买入一样，这种情况不足1%
            pass
        elif (last_row['macd'] > last_row['signal']) and (last_row['rsi'] < 30):
            # print(f"[趋势平仓] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                           round(indicatorEntity.tick_close, 3),
                           "buy"]
            positionEntity.trade_point_list.append(trade_point)


def strategy_c4_oscillation_boll(positionEntity, indicatorEntity, df):
    """震荡交易策略
    指标组合：KDJ + RSI + 布林带
    交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）
    """
    last_row = df.iloc[-1]

    # 布林带边界交易
    if last_row['close'] > last_row['boll_upper'] and last_row['rsi'] > 70:
        # print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | 触及上轨")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "sell"]
        positionEntity.trade_point_list.append(trade_point)
    elif last_row['close'] < last_row['boll_lower'] and last_row['rsi'] < 30:
        # print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | 触及下轨")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "buy"]
        positionEntity.trade_point_list.append(trade_point)


def strategy_c4_oscillation_kdj(positionEntity, indicatorEntity, df):
    """震荡交易策略
    指标组合：KDJ + RSI + 布林带
    交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）
    """
    last_row = df.iloc[-1]

    # KDJ交叉信号
    if df['k'].iloc[-1] > df['d'].iloc[-1] and df['k'].iloc[-2] <= df['d'].iloc[-2]:
        # print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ金叉")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "buy"]
        positionEntity.trade_point_list.append(trade_point)
    elif df['k'].iloc[-1] < df['d'].iloc[-1] and df['k'].iloc[-2] >= df['d'].iloc[-2]:
        # print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ死叉")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "sell"]
        positionEntity.trade_point_list.append(trade_point)


def strategy_c4_breakout(positionEntity, indicatorEntity, df):
    """突破交易策略
    指标组合：布林带 + ATR（真实波动范围）+ 成交量（VOL）
    交易策略：等待突破后回踩确认再进场
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    # 波动率放大条件
    volatility_cond = last_row['atr'] > df['atr'].rolling(20).mean().iloc[-1] * 1.2

    # 向上突破
    if last_row['close'] > last_row['boll_upper'] and \
            last_row['volume'] > prev_row['volume'] * 1.5 and \
            volatility_cond:
        # print(f"[突破做多] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "buy"]
        positionEntity.trade_point_list.append(trade_point)

    # 向下突破
    elif last_row['close'] < last_row['boll_lower'] and \
            last_row['volume'] > prev_row['volume'] * 1.5 and \
            volatility_cond:
        # print(f"[突破做空] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "sell"]
        positionEntity.trade_point_list.append(trade_point)


def strategy_c4_reversal(positionEntity, indicatorEntity, df):
    """趋势反转策略
    指标组合：MACD + OBV（能量潮）+ 资金流向（MFI）
    交易策略：确认趋势反转信号后介入，不抄底/摸顶
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # MACD反转信号
    macd_reversal = (last_row['histogram'] > 0) and (prev_row['histogram'] < 0)

    # OBV背离检测
    # obv_divergence = (last_row['close'] < prev_row['close']) and (last_row['obv'] > prev_row['obv'])
    obv_divergence = last_row['obv'] > prev_row['obv']
    # (条件太严了，没信号爆出，于是不要求价格必须下降，只要求OBV上升)

    # 均线交叉
    ema_cross = (last_row['ema10'] > last_row['ema60']) and (prev_row['ema10'] <= prev_row['ema60'])

    if macd_reversal and obv_divergence and ema_cross:  #
        # print(f"[反转买入] {last_row.name} | 价格: {last_row['close']:.2f} | OBV: {last_row['obv'] / 1e6:.2f}M")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "buy"]
        positionEntity.trade_point_list.append(trade_point)

    elif ((last_row['histogram'] < 0) and (prev_row['histogram'] > 0) and
          # (last_row['close'] > prev_row['close']) and 条件太严了
          (last_row['obv'] < prev_row['obv']) and
          (last_row['ema10'] < last_row['ema60']) and (prev_row['ema10'] >= prev_row['ema60'])
    ):
        # print(f"[反转卖出] {last_row.name} | 价格: {last_row['close']:.2f} | OBV: {last_row['obv'] / 1e6:.2f}M")
        trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                       round(indicatorEntity.tick_close, 3),
                       "sell"]
        positionEntity.trade_point_list.append(trade_point)
