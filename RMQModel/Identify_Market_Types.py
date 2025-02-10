import pandas as pd
import numpy as np
import scipy.signal as signal
import RMQData.Indicator as RMQIndicator
import Identify_Market_Types_Helper as IMTHelper


def label_market_condition(df):
    # 指标计算（保持不变）
    # df = self._calculate_indicators(df)

    # 计算所有指标
    df = IMTHelper.calculate_indicators(df)

    # 计算原始概率
    raw_trend = calculate_trend_probability(df)
    raw_range = calculate_range_probability(df)
    raw_reversal = calculate_reversal_probability(df)
    raw_breakout_tmp = calculate_breakout_probability(df)

    raw_breakout = raw_breakout_tmp['value']  # direction 值为 up down 代表突破方向

    # 统一Sigmoid处理平滑处理
    """
    公式：使用Sigmoid函数将原始趋势概率映射到[0,1]区间：
    效果：当原始概率在0.5附近时输出平滑过渡，消除硬阈值突变
    
    center 概率中值点（其中突破行情更倾向高阈值）
    steepness 斜率（突破判断更严格）
    """
    trend_prob = IMTHelper.sigmoid(raw_trend, center=0.5, steepness=10)
    range_prob = IMTHelper.sigmoid(raw_range, center=0.5, steepness=8)
    reversal_prob = IMTHelper.sigmoid(raw_reversal, center=0.4, steepness=12)
    breakout_prob = IMTHelper.sigmoid(raw_breakout, center=0.6, steepness=15)

    # 动态权重分配（基于波动率）
    volatility = df['atr'].iloc[-1] / df['close'].iloc[-1]
    weights = {
        'trend': IMTHelper.trend_weight(volatility),
        'range': IMTHelper.range_weight(volatility),
        'reversal': 0.2,  # 固定权重
        'breakout': 0.2
    }

    # 加权融合
    weighted_probs = {
        'trend': trend_prob * weights['trend'],
        'range': range_prob * weights['range'],
        'reversal': reversal_prob * weights['reversal'],
        'breakout': breakout_prob * weights['breakout']
    }

    # 风险控制检查（新增）
    IMTHelper.risk_control_check(weighted_probs)

    # 归一化处理
    total = sum(weighted_probs.values()) + 1e-8  # 防止除零
    # 最终归一化概率
    final_probs = {k: v / total for k, v in weighted_probs.items()}

    return final_probs


def calculate_trend_probability(df):
    """
    综合趋势判断逻辑（支持多时间框架验证）
    返回趋势概率（0.0~1.0）

    实现亮点：
        多维度趋势验证体系：
            6大核心条件：ADX强度、均线排列、价格结构、MACD动量、多时间框架验证、回调幅度过滤
            采用加权积分机制（基础分+持续加分+波动率调整）
        智能参数调整：
            动态EMA周期（根据波动率自动调整）
            自适应波动率权重（高波动市场降低趋势概率权重
        多时间框架协同：
            周线EMA50方向验证
            4小时斐波那契关键位验证
            日线与周线趋势一致性检查
        复合过滤机制：
            斐波那契回撤率+ATR波动率双过滤
            MACD与均线方向协同验证
            价格结构有效性检查（连续高/低点）
    """
    # 基础参数
    trend_prob = 0.0
    adx_threshold = 25
    min_trend_bars = 5
    fib_retrace_threshold = 0.382  # 斐波那契回撤率
    atr_multiplier = 1.5

    # 计算核心指标
    # df = self._calculate_ema(df)
    # df = self._calculate_macd(df)
    # df = self._calculate_adx(df)
    # df = self._calculate_atr(df)

    # 1. ADX趋势强度判断
    adx = df['adx'].iloc[-1]
    plus_di = df['plus_di'].iloc[-1]
    minus_di = df['minus_di'].iloc[-1]

    # 2. 均线系统判断
    ema20 = df['ema20'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    ema200 = df['ema200'].iloc[-1]
    price = df['close'].iloc[-1]

    # 3. 价格结构分析
    highs = df['high'].values[-30:]  # 最近30根K线高点
    lows = df['low'].values[-30:]  # 最近30根K线低点

    # 4. 多时间框架验证
    weekly_ema50_dir = IMTHelper.get_weekly_ema50_direction(df)
    h4_key_level = IMTHelper.get_h4_fib_level(df)

    # ================ 核心判断逻辑 ================
    # 条件1：ADX趋势强度
    adx_condition = adx > adx_threshold and (plus_di > minus_di or plus_di < minus_di)

    # 条件2：均线排列方向
    if ema20 > ema50 and ema50 > ema200 and price > ema20:
        ma_condition = 1  # 多头排列
    elif ema20 < ema50 and ema50 < ema200 and price < ema20:
        ma_condition = -1  # 空头排列
    else:
        ma_condition = 0

    # 条件3：价格结构有效性
    if ma_condition == 1:  # 上涨趋势结构
        price_structure = IMTHelper.is_valid_uptrend(highs, lows)
    elif ma_condition == -1:  # 下跌趋势结构
        price_structure = IMTHelper.is_valid_downtrend(highs, lows)
    else:
        price_structure = False

    # 条件4：MACD动量验证
    macd_signal = df['macd'].iloc[-1] > df['signal'].iloc[-1] if ma_condition == 1 else \
        df['macd'].iloc[-1] < df['signal'].iloc[-1]

    # 条件5：多时间框架验证
    if ma_condition == -1:
        tf_condition = weekly_ema50_dir < 0 and df['high'].iloc[-1] < h4_key_level
    else:
        tf_condition = weekly_ema50_dir > 0 and df['low'].iloc[-1] > h4_key_level

    # 条件6：回调幅度过滤
    current_atr = df['atr'].iloc[-1]
    if ma_condition != 0:
        pullback = IMTHelper.check_pullback_trend(df, ma_condition, fib_retrace_threshold,
                                                  atr_multiplier * current_atr)
    else:
        pullback = False

    # ================ 概率综合计算 ================
    # 基础得分
    if adx_condition:
        trend_prob += 0.3
    if ma_condition != 0:
        trend_prob += 0.2
    if price_structure:
        trend_prob += 0.2
    if macd_signal:
        trend_prob += 0.1
    if tf_condition:
        trend_prob += 0.1
    if pullback:
        trend_prob += 0.1

    # 时间持续性加分
    consecutive_days = IMTHelper.count_consecutive_days(df, ma_condition)
    if consecutive_days >= min_trend_bars:
        trend_prob += min(0.2, consecutive_days * 0.05)

    # 波动率调整
    volatility = df['atr'].mean() / df['close'].mean()  # 相对波动率
    if volatility > 0.15:  # 高波动市场
        trend_prob *= 0.8  # 降低趋势概率权重
    else:
        trend_prob *= 1.2  # 增强趋势概率权重

    return max(0.0, min(1.0, trend_prob))


def calculate_range_probability(df):
    """
    综合震荡行情判断逻辑
    返回震荡概率（0.0~1.0）

    六维震荡验证体系：
        A[ADX<20] --> B[震荡基础]
        C[布林带收口] --> B
        D[波动率收缩] --> B
        E[RSI中性] --> B
        F[OBV平衡] --> B
        G[成交量稳定] --> B
        B --> H[综合概率]
    动态学习机制：
        布林带带宽采用历史分位数比较（过去120日20%分位）
        OBV平衡检测使用线性回归斜率
        假突破检测窗口可配置（默认20日）
    风险感知系统 重大事件前降低震荡概率
    自适应性调整：
        成交量稳定性检测（5日/20日均量比）
        时间持续性加分（连续震荡天数越多概率越高）
        假突破频率惩罚机制

    各条件权重分配
    条件	权重	说明
    ADX	25%	趋势强度核心指标
    布林带	25%	波动率收缩直接证据
    波动率	20%	ATR与价格范围验证
    RSI	15%	市场情绪中性验证
    OBV	10%	资金流向平衡性验证
    成交量	5%	流动性稳定性验证

    优势：
        动态适应性：自动调整对成交量、波动率等参数的敏感度
        风险感知：识别假突破和事件驱动风险
        可解释性：每个条件的贡献度清晰可见
        可扩展性：模块化设计便于添加新判断条件
    """
    # 基础参数
    range_prob = 0.0
    adx_threshold = 20
    bandwidth_threshold = 0.06  # 6%
    atr_ratio_threshold = 0.7
    min_range_days = 5
    rsi_low, rsi_high = 30, 70

    # 计算必要指标
    # df = self._calculate_bollinger_bands(df)
    # df = self._calculate_rsi(df)
    # df = self._calculate_obv(df)
    # df = self._calculate_atr(df)

    # ================== 核心判断条件 ==================
    # 条件1：ADX趋势强度
    adx_condition = df['adx'].iloc[-min_range_days:].max() < adx_threshold

    # 条件2：布林带收口
    bandwidth = (df['boll_upper'].iloc[-1] - df['boll_lower'].iloc[-1]) / df['boll_mid'].iloc[-1]
    historical_bandwidth = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
    bandwidth_condition = (bandwidth < bandwidth_threshold) & \
                          (bandwidth < historical_bandwidth.rolling(120).quantile(0.2).iloc[-1])

    # 条件3：价格波动率收缩
    current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
    atr_20ma = df['atr'].rolling(20).mean().iloc[-1]
    volatility_condition = current_range < atr_20ma * atr_ratio_threshold

    # 条件4：RSI中性区间
    rsi_values = df['rsi'].iloc[-10:]
    rsi_condition = (rsi_values.min() > rsi_low) & (rsi_values.max() < rsi_high)

    # 条件5：OBV平衡
    obv_slope = IMTHelper.calculate_slope(df['obv'].iloc[-20:])
    obv_condition = abs(obv_slope) < 0.01  # OBV变动斜率小于1%

    # 条件6：成交量稳定
    volume_ma_ratio = df['volume'].iloc[-5:].mean() / df['volume'].rolling(20).mean().iloc[-1]
    volume_condition = 0.8 < volume_ma_ratio < 1.2

    # ================== 概率综合计算 ==================
    # 基础得分
    condition_weights = {
        'adx': 0.25,
        'bollinger': 0.25,
        'volatility': 0.2,
        'rsi': 0.15,
        'obv': 0.1,
        'volume': 0.05
    }

    if adx_condition:
        range_prob += condition_weights['adx']
    if bandwidth_condition:
        range_prob += condition_weights['bollinger']
    if volatility_condition:
        range_prob += condition_weights['volatility']
    if rsi_condition:
        range_prob += condition_weights['rsi']
    if obv_condition:
        range_prob += condition_weights['obv']
    if volume_condition:
        range_prob += condition_weights['volume']

    # 时间持续性加分
    consecutive_days = IMTHelper.count_consecutive_range_days(df)
    if consecutive_days >= min_range_days:
        range_prob += min(0.3, consecutive_days * 0.05)

    # 假突破惩罚项
    false_breakouts = IMTHelper.detect_false_breakouts(df)
    range_prob -= false_breakouts * 0.1

    # 事件驱动过滤
    if IMTHelper.has_upcoming_events(df):
        range_prob *= 0.5  # 重大事件前降低震荡概率

    return max(0.0, min(1.0, range_prob))


def calculate_reversal_probability(df):
    """
    综合趋势反转判断逻辑（三重验证体系）
    返回反转概率（0.0~1.0）
    """
    # 基础参数
    reversal_prob = 0.0
    volume_multiplier = 2.0  # 成交量激增倍数
    stop_loss_percent = 0.015  # 止损位偏移1.5%

    # 计算必要指标
    # df = self._calculate_macd(df)
    # df = self._calculate_obv(df)
    # df = self._calculate_mfi(df)
    # df = self._detect_price_patterns(df)

    # ================== 三重验证核心条件 ==================
    # 条件1：形态突破验证
    pattern_break = IMTHelper.check_pattern_break(df)

    # 条件2：量价背离验证
    divergence = IMTHelper.check_divergence(df)

    # 条件3：情绪极端验证
    sentiment = IMTHelper.check_extreme_sentiment(df)

    # ================== 辅助验证条件 ==================
    # 条件4：成交量激增
    volume_condition = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * volume_multiplier

    # 条件5：关键均线突破
    ma_condition = IMTHelper.check_ma_crossover(df)

    # 条件6：波动率扩张
    volatility_expansion = df['atr'].iloc[-1] > df['atr'].rolling(20).mean().iloc[-1] * 1.5

    # ================== 概率综合计算 ==================
    # 三重验证基础分（必须同时满足）
    if pattern_break['confirmed'] and divergence and sentiment:
        reversal_prob += 0.6
    elif pattern_break['confirmed'] and (divergence or sentiment):
        reversal_prob += 0.4

    # 辅助条件加分
    condition_weights = {
        'volume': 0.15,
        'ma': 0.1,
        'volatility': 0.05
    }
    if volume_condition:
        reversal_prob += condition_weights['volume']
    if ma_condition:
        reversal_prob += condition_weights['ma']
    if volatility_expansion:
        reversal_prob += condition_weights['volatility']

    # 形态强度调整
    if pattern_break['confirmed']:
        pattern_score = {
            'head_shoulder_top': 1.0,
            'head_shoulder_bottom': 1.0,
            'double_top': 0.8,
            'double_bottom': 0.8,
            'wedge': 0.7
        }.get(pattern_break['pattern_type'], 0.5)
        reversal_prob *= pattern_score

    # 止损空间惩罚项
    if IMTHelper.check_stop_loss_risk(df, stop_loss_percent):
        reversal_prob *= 0.7

    return max(0.0, min(1.0, reversal_prob))


def calculate_breakout_probability(df):
    """
    综合突破行情判断逻辑（支持多空双向验证）
    返回突破概率字典：{'up': 0.0~1.0, 'down': 0.0~1.0}

    动态概率调整：
        基础条件权重：突破幅度（30%）、成交量（20%）、持续站稳（30%）、ADX（10%）、布林带扩张（10%）
        回踩确认加成：有效回踩增加20%概率
        互斥性调整：双向突破概率总和超过100%时进行归一化
    真假突破过滤：
        量能验证：突破日成交量 > 5日均量200%
        持续验证：连续3根K线收于突破区间外
        动量验证：ADX指标上升且布林带开口扩大
    回踩机制检测：
        向上突破后回调不破前高
        向下突破后反弹不破前低
    """
    # 基础参数
    breakout_prob = {'up': 0.0, 'down': 0.0}
    atr_multiplier = 1.5
    volume_multiplier = 2.0
    confirmation_bars = 3
    lookback_period = 20

    # 计算必要指标
    # df = self._calculate_bollinger_bands(df)
    # df = self._calculate_atr(df)
    # df = self._calculate_adx(df)
    # df = self._calculate_volume_ma(df, periods=5)

    # ================== 核心突破条件 ==================
    # 突破方向检测
    up_break, down_break = IMTHelper.detect_initial_break(df, lookback_period)

    # 公共验证条件
    volume_condition = df['volume'].iloc[-1] > df['volume_ma5'].iloc[-1] * volume_multiplier
    adx_rising = df['adx'].iloc[-1] > df['adx'].rolling(5).mean().iloc[-1]
    bollinger_expanding = (df['boll_upper'].iloc[-1] - df['boll_lower'].iloc[-1]) > \
                          (df['boll_upper'].iloc[-lookback_period] - df['boll_lower'].iloc[-lookback_period]) * 1.2

    # ================== 向上突破验证 ==================
    if up_break:
        # 突破幅度验证
        break_magnitude = df['high'].iloc[-1] - df['high'].rolling(lookback_period).max().iloc[-2]
        magnitude_condition = break_magnitude > df['atr'].iloc[-1] * atr_multiplier

        # 持续站稳验证
        close_above = (df['close'].iloc[-confirmation_bars:] >
                       df['high'].rolling(lookback_period).max().iloc[-lookback_period - 1]).all()

        # 概率计算
        prob = 0.0
        if magnitude_condition:
            prob += 0.3
        if volume_condition:
            prob += 0.2
        if close_above:
            prob += 0.3
        if adx_rising:
            prob += 0.1
        if bollinger_expanding:
            prob += 0.1

        # 回踩加分项
        if IMTHelper.check_pullback(df, direction='up'):
            prob += 0.2

        breakout_prob['up'] = min(1.0, prob)

    # ================== 向下突破验证 ==================
    if down_break:
        # 突破幅度验证
        break_magnitude = df['low'].rolling(lookback_period).min().iloc[-2] - df['low'].iloc[-1]
        magnitude_condition = break_magnitude > df['atr'].iloc[-1] * atr_multiplier

        # 持续站稳验证
        close_below = (df['close'].iloc[-confirmation_bars:] <
                       df['low'].rolling(lookback_period).min().iloc[-lookback_period - 1]).all()

        # 概率计算
        prob = 0.0
        if magnitude_condition:
            prob += 0.3
        if volume_condition:
            prob += 0.2
        if close_below:
            prob += 0.3
        if adx_rising:
            prob += 0.1
        if bollinger_expanding:
            prob += 0.1

        # 回踩加分项
        if IMTHelper.check_pullback(df, direction='down'):
            prob += 0.2

        breakout_prob['down'] = min(1.0, prob)

    # 突破互斥性调整（不可能同时高概率双向突破）
    total = breakout_prob['up'] + breakout_prob['down']
    if total > 1.0:
        breakout_prob['up'] /= total
        breakout_prob['down'] /= total

    up_prob = breakout_prob['up']
    down_prob = breakout_prob['down']
    return {
        'value': max(up_prob, down_prob),
        'direction': 'up' if up_prob > down_prob else 'down'
    }


# 示例数据
data = {
    'time': pd.date_range(start='2023-01-01', periods=500, freq='D'),
    'open': np.random.random(500) * 100,
    'high': np.random.random(500) * 100 + 5,
    'low': np.random.random(500) * 100 - 5,
    'close': np.random.random(500) * 100,
    'volume': np.random.randint(1000, 10000, size=500)
}
df = pd.DataFrame(data)
df.set_index('time', inplace=True)

#读取CSV文件
#df = pd.read_csv('stock_data.csv', parse_dates=['time'], index_col='time')


# 标注每一段行情
labels = []
for i in range(len(df) - 250 + 1):
    window_df = df.iloc[i:i + 250].copy()
    print(window_df.shape)
    label = label_market_condition(window_df)
    print(window_df.shape)
    print(label)
    break
    #labels.append(label)

# 将标注结果添加到DataFrame
#df['label'] = pd.Series(labels, index=df.index[149:])
