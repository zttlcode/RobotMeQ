import pandas as pd
import matplotlib.pyplot as plt
from RMQStrategy import Identify_market_types_helper as IMTHelper
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools


class TradingStrategies:
    def __init__(self, df):
        self.df = df.copy()

    # ----------------- 核心策略 -----------------
    def trend_strategy(self):
        """趋势跟踪策略
        指标组合：均线（MA/EMA）+ MACD + ADX
        交易策略：顺势交易，逢低买入（上升趋势），逢高做空（下降趋势）
        """
        # 获取最新数据点
        last_row = self.df.iloc[-1]

        # 趋势方向判断
        trend_direction = 'up' if last_row['close'] > last_row['ema60'] else 'down'

        # 交易信号
        if trend_direction == 'up':
            # 多头策略
            if (last_row['macd'] > last_row['signal']) and (last_row['rsi'] < 30):
                print(f"[趋势买入] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            elif (last_row['macd'] < last_row['signal']) and (last_row['rsi'] > 70):
                print(f"[趋势卖出] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
        else:
            # 空头策略
            if (last_row['macd'] < last_row['signal']) and (last_row['rsi'] > 70):
                print(f"[趋势做空] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")
            elif (last_row['macd'] > last_row['signal']) and (last_row['rsi'] < 30):
                print(f"[趋势平仓] {last_row.name} | 价格: {last_row['close']:.2f} | RSI: {last_row['rsi']:.1f}")

    def oscillation_strategy(self):
        """震荡交易策略
        指标组合：KDJ + RSI + 布林带
        交易策略：在支撑位买入，在阻力位卖出（箱体交易策略）
        """
        last_row = self.df.iloc[-1]

        # 布林带边界交易
        if last_row['close'] > last_row['boll_upper'] and last_row['rsi'] > 70:
            print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | 触及上轨")
        elif last_row['close'] < last_row['boll_lower'] and last_row['rsi'] < 30:
            print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | 触及下轨")

        # KDJ交叉信号
        if self.df['k'].iloc[-1] > self.df['d'].iloc[-1] and self.df['k'].iloc[-2] <= self.df['d'].iloc[-2]:
            print(f"[震荡买入] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ金叉")
        elif self.df['k'].iloc[-1] < self.df['d'].iloc[-1] and self.df['k'].iloc[-2] >= self.df['d'].iloc[-2]:
            print(f"[震荡卖出] {last_row.name} | 价格: {last_row['close']:.2f} | KDJ死叉")

    def breakout_strategy(self):
        """突破交易策略
        指标组合：布林带 + ATR（真实波动范围）+ 成交量（VOL）
        交易策略：等待突破后回踩确认再进场
        """
        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]

        # 波动率放大条件
        volatility_cond = last_row['atr'] > self.df['atr'].rolling(20).mean().iloc[-1] * 1.2

        # 向上突破
        if last_row['close'] > last_row['boll_upper'] and \
                last_row['volume'] > prev_row['volume'] * 1.5 and \
                volatility_cond:
            print(
                f"[突破做多] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")

        # 向下突破
        elif last_row['close'] < last_row['boll_lower'] and \
                last_row['volume'] > prev_row['volume'] * 1.5 and \
                volatility_cond:
            print(
                f"[突破做空] {last_row.name} | 价格: {last_row['close']:.2f} | 成交量: {last_row['volume'] / 1e6:.2f}M")

    def reversal_strategy(self):
        """趋势反转策略
        指标组合：MACD + OBV（能量潮）+ 资金流向（MFI）
        交易策略：确认趋势反转信号后介入，不抄底/摸顶
        """
        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]

        # MACD反转信号
        macd_reversal = (last_row['histogram'] > 0) and (prev_row['histogram'] < 0)

        # OBV背离检测
        # obv_divergence = (last_row['close'] < prev_row['close']) and (last_row['obv'] > prev_row['obv'])
        obv_divergence = last_row['obv'] > prev_row['obv']
        # (条件太严了，没信号爆出，于是不要求价格必须下降，只要求OBV上升)

        # 均线交叉
        ema_cross = (last_row['ema10'] > last_row['ema60']) and (prev_row['ema10'] <= prev_row['ema60'])

        if macd_reversal and obv_divergence and ema_cross:  #
            print(f"[反转买入] {last_row.name} | 价格: {last_row['close']:.2f} | OBV: {last_row['obv'] / 1e6:.2f}M")

        elif ((last_row['histogram'] < 0) and (prev_row['histogram'] > 0) and
              # (last_row['close'] > prev_row['close']) and 条件太严了
              (last_row['obv'] < prev_row['obv']) and
              (last_row['ema10'] < last_row['ema60']) and (prev_row['ema10'] >= prev_row['ema60'])
        ):
            print(f"[反转卖出] {last_row.name} | 价格: {last_row['close']:.2f} | OBV: {last_row['obv'] / 1e6:.2f}M")


def calculate_indicators(df):
    """整合所有指标计算"""
    df = IMTHelper.calculate_ema(df)
    df = IMTHelper.calculate_macd(df)
    df = IMTHelper.calculate_atr(df)
    df = IMTHelper.calculate_bollinger_bands(df)
    df = IMTHelper.calculate_rsi(df)
    df = IMTHelper.calculate_obv(df)
    df = IMTHelper.calculate_kdj(df)

    return df


def plot_strategy(df):
    plt.figure(figsize=(16, 8))
    plt.plot(df['close'], label='Price')
    plt.plot(df['boll_upper'], linestyle='--', label='Boll Upper')
    plt.plot(df['boll_lower'], linestyle='--', label='Boll Lower')
    plt.scatter(df[df['signal'] == 'buy'].index, df[df['signal'] == 'buy']['close'], marker='^', color='g')
    plt.scatter(df[df['signal'] == 'sell'].index, df[df['signal'] == 'sell']['close'], marker='v', color='r')
    plt.legend()
    plt.show()


# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    allStockCode = pd.read_csv("../QuantData/a800_stocks.csv")
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['60'],
                                             'stock',
                                             1, 'A')
        for asset in assetList:
            # 读取CSV文件
            backtest_df_filePath = (RMTTools.read_config("RMQData", "backtest_bar")
                                    + "bar_"
                                    + asset.assetsMarket
                                    + "_"
                                    + asset.assetsCode
                                    + "_"
                                    + asset.barEntity.timeLevel
                                    + '.csv')
            df = pd.read_csv(backtest_df_filePath, encoding='utf-8', parse_dates=['time'], index_col="time")

            df = calculate_indicators(df)

            for i in range(0, len(df) - 120 + 1, 1):
                window_df = df.iloc[i:i + 120].copy()

                # 初始化策略引擎
                strategies = TradingStrategies(window_df)

                # 执行不同策略（根据实际行情类型调用）
                # strategies.trend_strategy()
                strategies.oscillation_strategy()
                # strategies.breakout_strategy()
                # strategies.reversal_strategy()

            break
