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
        使用前T根K线的特征预测后K根K线的标签
        推荐参数：T=10（特征窗口长度），K=3（标签滞后天数）
行情转换逻辑检验：
    禁止出现「趋势→趋势」连续标注（需有中间状态）
    突破后必须跟随趋势或反转

        指标组合：均线（MA/EMA）+ MACD + ADX
        交易策略：顺势交易，逢低买入（上升趋势），逢高做空（下降趋势）

        指标组合：KDJ + RSI + 布林带
        交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）

        指标组合：MACD + OBV（能量潮）+ 资金流向（MFI）
        交易策略：确认趋势反转信号后介入，不抄底/摸顶
        如果符合 突破 的标准（如：价格突破布林带上轨/下轨，成交量激增），标注为“突破”。

        指标组合：布林带 + ATR（真实波动范围）+ 成交量（VOL）
        交易策略：等待突破后回踩确认再进场
        价格突破关键支撑或阻力	布林带开口放大，ATR 激增，成交量剧增

        如果符合 反转 的标准（如：MACD 死叉/金叉，价格触及支撑位反转），标注为“反转”。
"""

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