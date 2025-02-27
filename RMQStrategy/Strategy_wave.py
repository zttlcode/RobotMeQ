from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import RMQData.Asset as RMQAsset
from RMQTool import Tools as RMTTools
from RMQStrategy import Identify_market_types_helper as IMTHelper


class WavePhaseAnalyzer:
    def __init__(self, df, swing_window=5):
        self.df = df
        self.swing_window = swing_window
        self.peak_indices = []
        self.trough_indices = []
        self.wave_labels = []
        self._label_waves()

    def _label_waves(self):
        """识别所有极值点"""
        highs = self.df['high'].values
        lows = self.df['low'].values

        # 寻找局部极值点
        peaks = argrelextrema(highs, np.greater, order=self.swing_window)[0]
        troughs = argrelextrema(lows, np.less, order=self.swing_window)[0]

        # 过滤有效极值点（幅度超过ATR的30%）
        atr = self.df['atr'].iloc[-1]

        self.peak_indices = peaks
        self.trough_indices = troughs

        """动态标记波浪结构（支持未完成状态）"""
        # all_points = sorted(self.peak_indices + self.trough_indices)
        all_points = sorted(np.concatenate([self.peak_indices, self.trough_indices]))
        self.wave_labels = []  # 格式: [('type', [indices], is_complete), ...]
        current_wave = []
        wave_type = None

        for i in range(1, len(all_points)):
            prev_idx = all_points[i - 1]
            curr_idx = all_points[i]

            # 判断当前段是否为推动浪
            is_impulse = self._is_impulse_wave(prev_idx, curr_idx)
            new_wave_type = 'Impulse' if is_impulse else 'Corrective'

            # 开始新波浪或延续当前波浪
            if not current_wave or wave_type != new_wave_type:
                if current_wave:
                    self._finalize_wave(current_wave, wave_type)
                current_wave = [prev_idx, curr_idx]
                wave_type = new_wave_type
            else:
                current_wave.append(curr_idx)

            # 实时验证并标记可能完成的波浪
            if wave_type == 'Impulse' and len(current_wave) >= 5:
                if self._validate_impulse(current_wave[:5]):
                    self.wave_labels.append(('Impulse', current_wave[:5], True))
                    current_wave = current_wave[4:]  # 允许重叠检测
            elif wave_type == 'Corrective' and len(current_wave) >= 3:
                if self._validate_corrective(current_wave[:3]):
                    self.wave_labels.append(('Corrective', current_wave[:3], True))
                    current_wave = current_wave[2:]

        # 处理最后未完成的波浪
        if current_wave:
            self.wave_labels.append((wave_type, current_wave, False))

    def _is_impulse_wave(self, start_idx, end_idx):
        """判断是否为推动浪段"""
        # 价格运动方向
        is_up = self.df['high'].iloc[end_idx] > self.df['high'].iloc[start_idx]
        is_down = self.df['low'].iloc[end_idx] < self.df['low'].iloc[start_idx]

        # 波动幅度验证
        price_change = abs(self.df['close'].iloc[end_idx] - self.df['close'].iloc[start_idx])
        atr = self.df['atr'].iloc[-1]

        # 排除横向波动
        return (is_up or is_down) and (price_change > atr * 0.5)

    def _validate_impulse(self, wave_points):
        """验证推动浪结构有效性"""
        # 浪2不能回撤浪1的100%
        wave1_retrace = self._fib_retracement(wave_points[0], wave_points[1], wave_points[2])
        if wave1_retrace > 0.618:
            # print("浪2回撤幅度过大")
            return False

        # 浪4不能进入浪1价格区间
        if self._check_price_overlap(wave_points[0], wave_points[1],
                                     wave_points[3], wave_points[4]):
            # print("浪4进入浪1价格区间")
            return False

        # 浪3不能是最短的一浪
        wave_lengths = [self._wave_length(wave_points[i], wave_points[i + 1])
                        for i in range(4)]
        if wave_lengths[2] == min(wave_lengths):
            # print("浪3不能是最短推动浪")
            return False
        return True

    def _validate_corrective(self, wave_points):
        """验证调整浪结构有效性"""
        # C浪必须超越A浪终点
        a_end = wave_points[1]
        c_end = wave_points[2]
        if self.df['low'].iloc[c_end] > self.df['low'].iloc[a_end]:
            # print("C浪未超越A浪区间")
            return False
        return True

    def _fib_retracement(self, start, peak, end):
        """计算斐波那契回撤比例"""
        move = self.df['high'].iloc[peak] - self.df['low'].iloc[start]
        retrace = self.df['high'].iloc[peak] - self.df['low'].iloc[end]
        return retrace / move

    def get_current_phase(self):
        """获取当前所处的子浪阶段"""
        #last_index = self.df.index[-1]
        last_index = len(self.df) - 1
        """生成基于波浪理论的交易信号"""
        signals = []
        # 优先检查未完成的波浪
        for label in reversed(self.wave_labels):
            wave_type, points, is_complete = label
            start = points[0]
            end = points[-1] if is_complete else last_index

            if start <= last_index <= end:
                # 推动浪阶段判断
                if wave_type == 'Impulse':

                    # 推动浪第二浪结束信号
                    if len(points) >= 2 and is_complete:
                        wave2_end = points[1]
                        # retrace = self._fib_retracement(points[0], points[1], points[2])
                        # if 0.382 <= retrace <= 0.618:
                        signals.append({
                            'type': 'BUY',
                            'description': 'Wave 2 Pullback Completion',
                            'index': wave2_end,
                            'time': self.df['close'].iloc[wave2_end+5],
                            'price': self.df['close'].iloc[wave2_end]
                        })
                        print(signals)

                    subwaves = self._detect_subwaves(points)
                    current_sub = self._find_current_subwave(subwaves, last_index)
                    if current_sub:
                        # return f"Impulse-Wave {current_sub}{' (Developing)' if not is_complete else ''}"
                        return current_sub if not is_complete else ''
                    else:
                        return None

                # 调整浪阶段判断
                elif wave_type == 'Corrective':
                    if len(points) <= 3:
                        letters = ['A', 'B', 'C'][:len(points)]
                        current_sub = letters[-1] if is_complete else letters[len(points) - 1]
                        return f"Corrective-Wave {current_sub}{' (Developing)' if not is_complete else ''}"
                    else:
                        return None

        return "Market in Transition"

    def _detect_subwaves(self, points):
        """识别推动浪内部的子浪结构（例如浪3中的小5浪）"""
        subwaves = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            if self._is_impulse_wave(start, end):
                subwaves.append((start, end, 'Impulse'))
            else:
                subwaves.append((start, end, 'Corrective'))
        return subwaves

    def _check_price_overlap(self, wave1_start, wave1_end, wave4_start, wave4_end):
        """检查两段价格区间是否重叠"""
        # 获取浪1的最高价和最低价
        wave1_high = self.df['high'].iloc[wave1_end]
        wave1_low = self.df['low'].iloc[wave1_start]

        # 获取浪4的最高价和最低价
        wave4_high = self.df['high'].iloc[wave4_start]
        wave4_low = self.df['low'].iloc[wave4_end]

        # 判断区间重叠：浪1的高点 >= 浪4低点 且 浪4高点 >= 浪1低点
        return (wave1_high >= wave4_low) and (wave4_high >= wave1_low)

    def _wave_length(self, start_idx, end_idx):
        """计算波浪段的长度（价格差）"""
        # 确定极值点类型
        is_start_peak = start_idx in self.peak_indices
        is_end_trough = end_idx in self.trough_indices

        if is_start_peak and is_end_trough:
            # 下跌段长度：前高 - 后低
            return self.df['high'].iloc[start_idx] - self.df['low'].iloc[end_idx]
        else:
            # 上涨段长度：后高 - 前低
            return self.df['high'].iloc[end_idx] - self.df['low'].iloc[start_idx]

    def _finalize_wave(self, current_wave, wave_type):
        """完成当前波浪的最终验证和存储"""
        if len(current_wave) < 2:
            return  # 忽略无效波浪段

        # 推动浪最小验证
        if wave_type == 'Impulse':
            # 至少需要3个点才能构成初步推动结构
            if len(current_wave) >= 3:
                # 验证基本推动结构
                if self._validate_impulse_basic(current_wave):
                    self.wave_labels.append(
                        (wave_type, current_wave.copy(), False)
                    )

        # 调整浪最小验证
        elif wave_type == 'Corrective':
            # 至少需要2个点构成调整基础
            if len(current_wave) >= 2:
                if self._validate_corrective_basic(current_wave):
                    self.wave_labels.append(
                        (wave_type, current_wave.copy(), False)
                    )

    def _validate_impulse_basic(self, points):
        """推动浪基础验证（适用于未完成结构）"""
        # 浪2回撤不超过浪1的100%
        if len(points) >= 2:
            retrace = self._fib_retracement(points[0], points[1], points[2])
            if retrace > 1.0:
                # print("浪2完全回撤浪1")
                return False
        return True

    def _validate_corrective_basic(self, points):
        """调整浪基础验证"""
        # A浪和B浪的波动方向验证
        if len(points) >= 2:
            price_change = self.df['close'].iloc[points[1]] - self.df['close'].iloc[points[0]]
            if abs(price_change) < self.df['atr'].iloc[-1] * 0.2:
                # print("调整浪幅度不足")
                return False
        return True

    def _find_current_subwave(self, subwaves, last_index):
        """精确识别当前子浪阶段"""
        # subwaves格式: [(start_idx, end_idx, wave_type), ...]
        current_sub = 1
        max_confirmation = 0

        # 多尺度验证（从大级别到小级别）
        for scale in [1, 0.618, 0.382]:  # 不同时间尺度验证
            confirmed_waves = []
            for i, (start, end, w_type) in enumerate(subwaves):
                scaled_duration = (end - start) * scale
                if last_index >= start and last_index <= start + scaled_duration:
                    confirmed_waves.append(i + 1)  # 记录可能处于的子浪

            # 选择最可能子浪（多数尺度确认）
            if len(confirmed_waves) > max_confirmation:
                current_sub = Counter(confirmed_waves).most_common(1)[0][0]
                max_confirmation = len(confirmed_waves)

        # 方向确认（避免将回调误判为推进）
        if not subwaves:
            return None
        last_confirmed = subwaves[current_sub - 1][1]
        if current_sub < len(subwaves):
            next_start = subwaves[current_sub][0]
            current_close = self.df['close'].iloc[last_index]
            prev_close = self.df['close'].iloc[last_confirmed]

            # 方向一致性检查
            if (current_sub % 2 == 1):  # 奇数浪应为推动方向
                if current_close < prev_close:
                    current_sub = max(1, current_sub - 1)
            else:  # 偶数浪应为调整方向
                if current_close > prev_close:
                    current_sub = max(1, current_sub - 1)
        return current_sub


def plt_wave_labels(analyzer):
    # 可视化波浪标记
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 8))
    plt.plot(df['close'], label='Price')

    # 标记推动浪
    for label in analyzer.wave_labels:
        if label[0] == 'Impulse':
            points = label[1]
            plt.scatter(df.index[points], df['close'].iloc[points],
                        color='red', marker='^', s=100)
            for i in range(len(points) - 1):
                plt.plot(df.index[points[i]:points[i + 1]],
                         df['close'].iloc[points[i]:points[i + 1]],
                         'r--', alpha=0.5)

    # 标记调整浪
    for label in analyzer.wave_labels:
        if label[0] == 'Corrective':
            points = label[1]
            plt.scatter(df.index[points], df['close'].iloc[points],
                        color='blue', marker='v', s=100)
            for i in range(len(points) - 1):
                plt.plot(df.index[points[i]:points[i + 1]],
                         df['close'].iloc[points[i]:points[i + 1]],
                         'b--', alpha=0.5)

    plt.title("Elliott Wave Analysis")
    plt.legend()
    plt.show()


# 使用示例
if __name__ == "__main__":
    allStockCode = pd.read_csv("../QuantData/a800_stocks.csv", dtype={'code': str})
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
            df = IMTHelper.calculate_atr(df)

            for i in range(0, len(df) - 120 + 1, 1):
                window_df = df.iloc[i:i + 120].copy()

                analyzer = WavePhaseAnalyzer(window_df)
                if analyzer.get_current_phase() == 2:
                    print(111)

                # plt_wave_labels(analyzer)

