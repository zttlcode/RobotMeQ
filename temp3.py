""""""
"""
行情类型	价格行为特征	指标特征
查看一段时间（例如：30天、60天）的市场行情。
    MACD(12,26,9)：快线、慢线、柱状线
    RSI(14)
    ADX(14) +DI/-DI
    布林带：上轨、中轨、下轨、带宽
    成交量：当日量、5日均量比
    ATR(14)
        # 特征窗口配置示例
        feature_windows = {
            'price_high': 20,    # 20日最高价
            'price_low': 20,     # 20日最低价 
            'volume_ma': 5,      # 5日成交量均线
            'adx_trend': 3       # ADX连续达标天数
        }
        使用前T根K线的特征预测后K根K线的标签
        推荐参数：T=10（特征窗口长度），K=3（标签滞后天数）
行情转换逻辑检验：
    禁止出现「趋势→趋势」连续标注（需有中间状态）
    突破后必须跟随趋势或反转

趋势行情	（Trending Market） 价格持续上涨/下跌	顺势交易	MA、MACD、ADX
        🔹 特点： 价格持续上升或下降，呈现 单边运动，且波动幅度大。
        🔹 分类： 上升趋势（牛市） / 下降趋势（熊市）
        价格持续上涨或下跌	均线多头排列，MACD 强势，ADX 高
        高点和低点 不断抬高（上涨趋势） 或 不断降低（下跌趋势）
        价格运行在 均线上方（上涨） 或 均线下方（下跌）
        MACD、ADX 等趋势指标呈现 强趋势信号
        ✅ 适用策略：趋势跟随

        指标组合：均线（MA/EMA）+ MACD + ADX
        交易策略：顺势交易，逢低买入（上升趋势），逢高做空（下降趋势）

        如果符合 趋势 的标准（如：ADX > 25，MACD 强势），标注为“趋势”。


        规则：
        有效下跌趋势：连续出现
        更低的高点（Lower
        Highs） 和
        更低的低点（Lower
        Lows）
        反弹无效条件：反弹幅度 < 前一波段跌幅的38
        .2 %（斐波那契回撤位）
        
        def is_valid_downtrend(highs, lows, threshold=0.382):
            for i in range(2, len(highs)):
                prev_high = highs[i - 2]
                current_high = highs[i - 1]
                next_high = highs[i]
                # 检查是否形成更低的高点序列
                if not (current_high < prev_high and next_high < current_high):
                    return False
                # 计算反弹幅度是否超过阈值
                drawdown = (prev_high - lows[i - 1]) / prev_high
                rebound = (highs[i] - lows[i - 1]) / lows[i - 1]
                if rebound > drawdown * threshold:
                    return False
            return True


        二、指标参数优化
        自适应均线系统
        动态均线周期
        使用EMA（指数移动平均）而非SMA（简单移动平均），减少滞后
        
        
        def dynamic_ma_period(volatility):
            # 波动率越高，均线周期越长（过滤噪音）
            if volatility < 0.1:  # 年化波动率<10%
                return 20
            elif volatility < 0.3:  # 10%-30%
                return 50
            else:  # >30%
                return 100
        
        
        ADX趋势强度过滤
        增强规则：
        ADX(14) > 25
        且 + DI < -DI（下跌趋势）
        当ADX从高位回落但未跌破20时，仍维持趋势判断
        波动率阈值
        通过ATR(14)
        设定反弹过滤阀值：
        
        def is_noise_rebound(current_high, prev_low, atr):
            rebound_size = current_high - prev_low
            # 反弹幅度小于1.5倍ATR视为噪音
            return rebound_size < 1.5 * atr



        单边延续性：价格沿均线方向持续运行，回调幅度不超过前一波段的38.2%（斐波那契关键位）。
        动量验证：ADX值>25确认趋势强度，MACD柱状线持续扩张。
        策略优化
            入场时机：
            均线回踩：EMA20/50附近出现看涨吞没/K线缩量止跌。
            趋势线反弹：连接波段低点/高点形成的趋势线支撑阻力有效。
            风险管理：
            移动止损：追踪止损设置为ATR(14)的2倍，或前低/高下方1%。
            仓位分档：首次入场50%，突破前高/低加仓30%，剩余资金应急。

        趋势行情中MACD金叉、均线多头排列、RSI超买可能同步触发
        确认条件（需同时满足）：
            ADX(14) > 25 且持续3天以上
            价格连续创新高（上涨趋势）或新低（下跌趋势）持续≥5根K线
            EMA20与EMA50金叉（上涨）/死叉（下跌）且间距扩大
        标注方法：
            当条件满足时，以当前K线收盘价向前标注最近符合趋势的5-20根K线
            示例：若2023-10-05确认上升趋势成立，则标注10/01-10/05为趋势行情

    三、多时间框架验证
    1. 周线趋势锚定
        日线反弹时，检查周线是否维持下跌结构：
            周线EMA50方向向下
            周线未出现连续两根阳线收盘价高于前周高点
    2. 4小时关键位验证
        日线反弹若未突破4小时级别的关键阻力位（如斐波那契38.2%），视为无效


震荡行情	Range-Bound Market / Sideways Market） 价格在区间内反复	高抛低吸	RSI、KDJ、布林带
        🔹 特点： 价格在 区间内反复波动，无明确趋势，高点与低点在某个范围内反复出现。
        价格在区间内反复	布林带收窄，RSI 在 30-70 之间，平均趋向指数 ADX 低
        价格 围绕某个中轴 运行，突破失败
        KDJ、RSI、CCI 等 震荡指标在 30-70 区间内徘徊
        成交量波动小，资金流入流出不明显
        ✅ 适用策略：区间交易 / 高抛低吸

        指标组合：KDJ + RSI + 布林带
        交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）

        如果符合 震荡 的标准（如：ADX < 20，RSI 在 30-70 区间内），标注为“震荡”。

        波动收缩：布林带带宽（Bandwidth）降至6个月低位，ATR值低于均值30%。
        多空平衡：OBV指标持平，显示资金无明确方向偏好。
        策略进阶
            区间边缘交易：
                保守型：价格触及区间上沿（RSI>70）且出现看跌Pin Bar时做空。
                激进型：假突破后反向交易，例如突破前高后快速回落至区间内。
            期权策略：卖出跨式期权（Sell Straddle），赚取时间价值衰减收益。
        风险警示
            假突破陷阱：突破时若成交量未达20日均量2倍，需警惕诱多/诱空。
            事件驱动：财报/政策发布前避免区间交易，防止波动率骤增。

        震荡行情中KDJ超卖、布林带收口、成交量萎缩呈现协同性

        确认条件：
            ADX(14) < 20 持续5天
            布林带带宽（Bandwidth）< 6%
            最高价-最低价范围 < 过去20日ATR均值的70%
        标注方法：
            从条件首次满足的K线开始持续标注，直至任一条件被打破
            示例：2023-08-01至2023-08-15期间持续满足震荡条件，则标注该区间

趋势反转行情（Reversal Market） 价格突破趋势线	反转交易	MACD、OBV、双顶/双底
        🔹 特点： 价格 从趋势行情转向新的趋势，通常发生在重要支撑/阻力位。	
        价格趋势发生变化	MACD 变向，双顶/双底形态，成交量异常
        经典的 双顶/双底、头肩顶/头肩底 形态出现
        MACD 发生死叉/金叉，均线趋势翻转
        成交量异常放大，资金开始大量流入或流出
        ✅ 适用策略：反转交易 / 右侧交易

        指标组合：MACD + OBV（能量潮）+ 资金流向（MFI）
        交易策略：确认趋势反转信号后介入，不抄底/摸顶

        如果符合 突破 的标准（如：价格突破布林带上轨/下轨，成交量激增），标注为“突破”。

        确认信号三重验证
            形态完成：头肩顶颈线破位，回抽确认失败。
            量价背离：价格新高但MACD柱状线走低，OBV同步下跌。
            情绪极端：散户多头持仓比例>70%（如Coinbase永续合约数据）。
        交易策略
            分阶段建仓：
                第一仓（20%）：形态破位+收盘价确认。
                第二仓（30%）：回测颈线/趋势线反弹失败。
            止损设置：置于形态最高点/最低点外侧1.5%。

        确认条件：
            出现经典反转形态（头肩顶/底、双顶/底）且突破颈线
            MACD柱状线与价格出现背离（价格新高而MACD未新高）
            RSI(14) > 70（顶部反转）或 < 30（底部反转）
        标注方法：
            在颈线突破确认时向前追溯标注形态构筑期
            示例：头肩顶形态在2023-07-20破颈线，则标注07/10-07/20为反转行情


突破行情（Breakout Market） 价格突破震荡区间	突破交易	布林带、ATR、成交量
        🔹 特点： 价格在震荡整理后，突然突破重要阻力或支撑位，进入新趋势。
        突破前波动收窄，成交量减少（挤压效应）
        突破时成交量激增，趋势动量指标（ADX、ATR）拉升
        突破支撑/阻力后，价格往往出现 回踩确认 再继续上涨/下跌
        ✅ 适用策略：突破交易 / 顺势交易

        指标组合：布林带 + ATR（真实波动范围）+ 成交量（VOL）
        交易策略：等待突破后回踩确认再进场
        价格突破关键支撑或阻力	布林带开口放大，ATR 激增，成交量剧增

        如果符合 反转 的标准（如：MACD 死叉/金叉，价格触及支撑位反转），标注为“反转”。

        真/假突破鉴别
            量能验证：突破日成交量需高于前5日均量50%以上。
            持续验证：连续3根K线收盘站稳阻力位上方（如周线级别突破）。
        策略执行
            突破类型处理：
                延续突破：上升三角整理后的顺势加仓，目标位=形态高度1:1投射。
                反转突破：下降通道上破配合VIX指数骤降，转多头趋势。
            止盈技巧：
                分档止盈：50%仓位在1:1目标位了结，剩余部分追踪止盈。

        确认条件：
            价格突破过去20日最高价/最低价，且突破幅度 > 1.5倍ATR(14)
            突破当日成交量 > 前5日平均成交量200%
            突破后连续3根K线收盘价保持在突破区间外
        标注方法：
            标注突破确认日（第3根K线收盘）及之后3根K线
            示例：若价格在2023-09-01突破箱体，09/04确认有效突破，则标注09/04-09/07

终极风控原则
    单笔风险<2%：无论多强信号，止损设置保证单次亏损不超过总资金2%。
    跨周期验证：日线突破需周线趋势支持，避免陷入局部噪音。
    情绪过滤器：当CNN恐惧贪婪指数>75，强制降低仓位至50%以下。

"""

import pandas as pd
import numpy as np
import talib


def calculate_indicators(df):
    """ 计算技术指标 """
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)  # 50日均线
    df['MA200'] = talib.SMA(df['close'], timeperiod=200)  # 200日均线
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])  # MACD
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # RSI
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # 波动率
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # 趋势强度
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['close'], timeperiod=20)
    return df


def label_market_condition(df):
    """ 自动标注市场类型 """
    labels = []
    for i in range(len(df)):
        if df['ADX'][i] > 25 and df['MACD'][i] > df['MACD_signal'][i]:
            labels.append('trend')  # 趋势
        elif df['ADX'][i] < 20 and 30 < df['RSI'][i] < 70:
            labels.append('range')  # 震荡
        elif df['close'][i] > df['UpperBand'][i] or df['close'][i] < df['LowerBand'][i]:
            labels.append('breakout')  # 突破
        elif df['MACD'][i] < df['MACD_signal'][i] and df['MACD'][i - 1] > df['MACD_signal'][i - 1]:
            labels.append('reversal')  # 反转
        else:
            labels.append('range')  # 默认震荡
    df['Market_Label'] = labels
    return df


def detect_market_condition(df):
    """ 识别市场类型 """
    latest = df.iloc[-1]  # 取最新数据
    trend_strength = latest['ADX']

    # 1️⃣ 趋势行情（ADX > 25 且 MACD 强势）
    if trend_strength > 25 and latest['MACD'] > latest['MACD_signal']:
        return "trend"

    # 2️⃣ 震荡行情（ADX < 20 且 RSI 介于 30-70）
    elif trend_strength < 20 and 30 < latest['RSI'] < 70:
        return "range"

    # 3️⃣ 突破行情（价格突破布林带上轨或下轨，成交量放大）
    elif latest['close'] > latest['Bollinger_Upper'] or latest['close'] < latest['Bollinger_Lower']:
        return "breakout"

    # 4️⃣ 反转行情（MACD 死叉/金叉，ADX 下降）
    elif latest['MACD'] < latest['MACD_signal'] and df.iloc[-2]['ADX'] > latest['ADX']:
        return "reversal"

    # 默认震荡
    return "range"


def select_trading_strategy(market_condition):
    """ 根据市场行情选择交易策略 """
    if market_condition == "trend":
        return "顺势交易策略（均线+MACD）"
    elif market_condition == "range":
        return "震荡交易策略（KDJ+RSI）"
    elif market_condition == "breakout":
        return "突破交易策略（布林带+ATR）"
    elif market_condition == "reversal":
        return "反转交易策略（MACD+OBV）"
    else:
        return "未识别市场类型"