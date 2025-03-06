import pandas as pd
import matplotlib.pyplot as plt
from RMQStrategy import Identify_market_types_helper as IMTHelper
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools
import time


def strategy_c4_trend(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    """趋势跟踪策略
    指标组合：均线（MA/EMA）+ MACD + ADX
    交易策略：顺势交易，逢低买入（上升趋势），逢高做空（下降趋势）
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            # 获取最新数据点
            last_row = windowDF_calIndic.iloc[-1]

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
                    # 设置推送消息
                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                             + "-"
                                             + indicatorEntity.IE_assetsCode,
                                             indicatorEntity,
                                             None,
                                             "目前顶背离+KDJ死叉")
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
                    # 推送消息
                    strategy_result.send_msg(indicatorEntity.IE_assetsName
                                             + "-"
                                             + indicatorEntity.IE_assetsCode,
                                             indicatorEntity,
                                             None,
                                             "目前底背离+KDJ金叉")


def strategy_c4_oscillation_boll(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    """震荡交易策略
    指标组合：KDJ + RSI + 布林带
    交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            last_row = windowDF_calIndic.iloc[-1]

            # 布林带边界交易
            if last_row['close'] > last_row['boll_upper'] and last_row['rsi'] > 70:
                # print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | 触及上轨")
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
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
            elif last_row['close'] < last_row['boll_lower'] and last_row['rsi'] < 30:
                # print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | 触及下轨")
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
                                         None,
                                         "目前底背离+KDJ金叉")


def strategy_c4_oscillation_kdj(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    """震荡交易策略
    指标组合：KDJ + RSI + 布林带
    交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            last_row = windowDF_calIndic.iloc[-1]

            # KDJ交叉信号
            if (windowDF_calIndic['k'].iloc[-1] > windowDF_calIndic['d'].iloc[-1] and
                    windowDF_calIndic['k'].iloc[-2] <= windowDF_calIndic['d'].iloc[-2]):
                # print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ金叉")
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
                                         None,
                                         "目前底背离+KDJ金叉")
            elif (windowDF_calIndic['k'].iloc[-1] < windowDF_calIndic['d'].iloc[-1] and
                  windowDF_calIndic['k'].iloc[-2] >= windowDF_calIndic['d'].iloc[-2]):
                # print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ死叉")
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
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


def strategy_c4_breakout(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    """突破交易策略
    指标组合：布林带 + ATR（真实波动范围）+ 成交量（VOL）
    交易策略：等待突破后回踩确认再进场
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            last_row = windowDF_calIndic.iloc[-1]
            prev_row = windowDF_calIndic.iloc[-2]

            # 波动率放大条件
            volatility_cond = last_row['atr'] > windowDF_calIndic['atr'].rolling(20).mean().iloc[-1] * 1.2

            # 向上突破
            if last_row['close'] > last_row['boll_upper'] and \
                    last_row['volume'] > prev_row['volume'] * 1.5 and \
                    volatility_cond:
                # print(f"[突破做多] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")
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
                                         None,
                                         "突破做多")

            # 向下突破
            elif last_row['close'] < last_row['boll_lower'] and \
                    last_row['volume'] > prev_row['volume'] * 1.5 and \
                    volatility_cond:
                # print(f"[突破做空] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")
                trade_point = [indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                               round(indicatorEntity.tick_close, 3),
                               "sell"]
                positionEntity.trade_point_list.append(trade_point)
                # 设置推送消息
                strategy_result.send_msg(indicatorEntity.IE_assetsName
                                         + "-"
                                         + indicatorEntity.IE_assetsCode,
                                         indicatorEntity,
                                         None,
                                         "突破做空")


def strategy_c4_reversal(positionEntity, indicatorEntity, windowDF_calIndic, strategy_result):
    """趋势反转策略
    指标组合：MACD + OBV（能量潮）+ 资金流向（MFI）
    交易策略：确认趋势反转信号后介入，不抄底/摸顶
    """
    current_min = int(indicatorEntity.tick_time.strftime('%M'))
    if current_min % 5 == 0:  # 判断时间被5整除，如果是，说明bar刚更新，计算指标，否则不算指标；'%Y-%m-%d %H:%M'
        if current_min != indicatorEntity.last_cal_time:  # 说明bar刚更新，计算一次指标
            indicatorEntity.last_cal_time = current_min  # 更新锁

            last_row = windowDF_calIndic.iloc[-1]
            prev_row = windowDF_calIndic.iloc[-2]

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
                                         None,
                                         "目前底背离+KDJ金叉")

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
                # 设置推送消息
                strategy_result.send_msg(indicatorEntity.IE_assetsName
                                         + "-"
                                         + indicatorEntity.IE_assetsCode,
                                         indicatorEntity,
                                         None,
                                         "目前顶背离+KDJ死叉")
