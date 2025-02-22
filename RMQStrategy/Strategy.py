import numpy as np
import RMQData.Position as RMQPosition
import RMQData.Indicator as RMQIndicator
import RMQStrategy.Strategy_fuzzy as RMQSFuzzy
import RMQStrategy.Strategy_tea as RMQSTea
import RMQStrategy.Strategy_nature as RMQSNature
from RMQTool import Message
from RMQModel import Identify_market_types_helper as IMTHelper


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


def strategy(asset, strategy_result, IEMultiLevel, strategy_name):
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
    if strategy_name.startswith("c4_"):
        windowDF_calIndic = IMTHelper.calculate_ema(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_macd(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_atr(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_bollinger_bands(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_rsi(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_obv(windowDF_calIndic)
        windowDF_calIndic = IMTHelper.calculate_kdj(windowDF_calIndic)
    # -----------------------------------------------------------------------------------------------

    if strategy_name == "tea_conservative":
        # 保守派背离策略
        # 2%移动止损
        # 各级别参考日线KDJ，各自报信号时不累计背离次数
        RMQSTea.strategy_tea_conservative(positionEntity,
                                          indicatorEntity,
                                          windowDF_calIndic,
                                          barEntity.bar_num - 2,  # 比如时间窗口60，最后一条数据下标永远是59
                                          strategy_result,
                                          IEMultiLevel)
    elif strategy_name == "tea_radical":
        # 激进派背离策略
        # 无止损
        # 各级别参考日线KDJ，各自报信号时累计背离次数
        RMQSTea.strategy_tea_radical(positionEntity,
                                     indicatorEntity,
                                     windowDF_calIndic,
                                     barEntity.bar_num - 2,  # 比如时间窗口60，最后一条数据下标永远是59
                                     strategy_result,
                                     IEMultiLevel)
    elif strategy_name == "tea_radical_nature":
        # 激进派背离策略  修改：取消日线，不发消息
        RMQSNature.strategy_tea_radical(positionEntity,
                                        indicatorEntity,
                                        windowDF_calIndic,
                                        barEntity.bar_num - 2,  # 比如时间窗口60，最后一条数据下标永远是59
                                        strategy_result,
                                        IEMultiLevel)
    elif strategy_name == "fuzzy":
        # ride-mood策略
        # 2%移动止损
        # 增强版趋势跟随策略，反指率高达90%，经常小亏，偶尔大赚
        RMQSFuzzy.strategy_fuzzy(positionEntity,
                                 indicatorEntity,
                                 windowDF_calIndic,
                                 barEntity.bar_num - 1,  # 减了个实时价格，250变249，所以这里长度也跟着变成249
                                 strategy_result)
    elif strategy_name == "fuzzy_nature":
        # ride-mood策略
        # 2%移动止损
        # 增强版趋势跟随策略，反指率高达90%，经常小亏，偶尔大赚
        RMQSNature.strategy_fuzzy(positionEntity,
                                  indicatorEntity,
                                  windowDF_calIndic,
                                  barEntity.bar_num - 1,  # 减了个实时价格，250变249，所以这里长度也跟着变成249
                                  strategy_result)
    elif strategy_name == "c4_trend_nature":
        RMQSNature.strategy_c4_trend(positionEntity, indicatorEntity, windowDF_calIndic)
    elif strategy_name == "c4_oscillation_boll_nature":
        RMQSNature.strategy_c4_oscillation_boll(positionEntity, indicatorEntity, windowDF_calIndic)
    elif strategy_name == "c4_oscillation_kdj_nature":
        RMQSNature.strategy_c4_oscillation_kdj(positionEntity, indicatorEntity, windowDF_calIndic)
    elif strategy_name == "c4_breakout_nature":
        RMQSNature.strategy_c4_breakout(positionEntity, indicatorEntity, windowDF_calIndic)
    elif strategy_name == "c4_reversal_nature":
        RMQSNature.strategy_c4_reversal(positionEntity, indicatorEntity, windowDF_calIndic)
    else:
        print("未指定策略")
