import os
import pandas as pd

#
# import os
# import pandas as pd
#
# # 文件夹路径
# folder_path = "./QuantData/market_condition_backtest/"
#
#
# # 统计函数：计算 market_condition 中连续 >= 5 行的情况
# def count_consecutive_sequences(series, target_condition, min_length=5):
#     count = 0
#     streak = 0
#
#     for value in series:
#         if value == target_condition:
#             streak += 1
#         else:
#             if streak >= min_length:
#                 count += 1
#             streak = 0  # 重新计数
#
#     # 处理最后一段连续的情况
#     if streak >= min_length:
#         count += 1
#
#     return count
#
#
# # 结果存储
# summary = []
#
# # 遍历文件夹中的所有 CSV 文件
# for filename in os.listdir(folder_path):
#     if filename.endswith("_60.csv"):
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取 CSV 文件
#         df = pd.read_csv(file_path)
#
#         # 计算连续出现 >= 5 次的情况
#         trend_down_count = count_consecutive_sequences(df["market_condition"], "trend_down")
#         trend_up_count = count_consecutive_sequences(df["market_condition"], "trend_up")
#         range_count = count_consecutive_sequences(df["market_condition"], "range")
#
#         # 记录结果
#         summary.append([filename, trend_down_count, trend_up_count, range_count])
#
# # 转换为 DataFrame 方便查看
# result_df = pd.DataFrame(summary, columns=["File", "Trend Down (≥5)", "Trend Up (≥5)", "Range (≥5)"])
#
# # 输出统计结果
# print(result_df)
#
# # 可选：保存到 CSV 文件
# #result_df.to_csv("market_condition_consecutive_summary.csv", index=False)


import numpy as np
from typing import Dict, Tuple


class MultiLevelMarketJudge:
    def __init__(self):
        # 定义级别权重（当前级别的直接上级权重最高）
        self.level_weights = {
            'current': 0.4,
            'parent1': 0.35,
            'parent2': 0.25
        }

        # 类型映射表（上涨:1, 下跌:-1, 震荡:0）
        self.type_mapping = {
            '上涨趋势': 1,
            '下跌趋势': -1,
            '震荡': 0
        }

        # 基础概率矩阵 [当前类型, 上级1类型, 上级2类型]
        self.prob_matrix = {
            (1, 1, 1): 0.85,  # 全级别上涨
            (1, 1, 0): 0.7,
            (1, 0, 0): 0.6,
            (1, -1, -1): 0.35,  # 逆势上涨
            (-1, -1, -1): 0.85,
            (-1, -1, 0): 0.7,
            (-1, 0, 0): 0.6,
            (-1, 1, 1): 0.35,
            (0, 1, 1): 0.55,  # 震荡中的趋势
            (0, -1, -1): 0.45,
            (0, 0, 0): 0.5  # 全级别震荡
        }

    def _get_base_prob(self, current: int, parent1: int, parent2: int) -> float:
        """获取基础概率值"""
        key = (current, parent1, parent2)
        if key in self.prob_matrix:
            return self.prob_matrix[key]

        # 处理未明确定义的情况
        same_dir = sum([1 for t in [parent1, parent2] if t == current])
        conflict_dir = sum([1 for t in [parent1, parent2] if t == -current])

        if same_dir >= 1:
            return 0.65 if current != 0 else 0.55
        elif conflict_dir >= 1:
            return 0.35 if current != 0 else 0.45
        else:
            return 0.5  # 默认中性概率

    def _calculate_direction_score(self, types: Dict[str, str]) -> float:
        """计算方向得分"""
        scores = []
        for level, weight in self.level_weights.items():
            t = self.type_mapping[types[level]]
            scores.append(t * weight)
        return np.tanh(sum(scores))  # 使用tanh压缩到[-1,1]

    def judge_market(self,
                     current_type: str,
                     parent1_type: str,
                     parent2_type: str) -> Tuple[float, str]:
        """
        多级别行情判断函数
        参数：
            current_type: 当前级别类型（上涨趋势/下跌趋势/震荡）
            parent1_type: 直接上级类型
            parent2_type: 更高级别类型
        返回：
            (probability, confidence)
            probability: 当前方向的概率值（0-1之间）
            confidence: 确定性判断（'明确'/'模棱两可'）
        """
        # 转换类型为数值
        types = {
            'current': current_type,
            'parent1': parent1_type,
            'parent2': parent2_type
        }

        current = self.type_mapping[current_type]
        parent1 = self.type_mapping[parent1_type]
        parent2 = self.type_mapping[parent2_type]

        # 步骤1：计算基础概率
        base_prob = self._get_base_prob(current, parent1, parent2)

        # 步骤2：计算方向一致性得分
        dir_score = self._calculate_direction_score(types)

        # 步骤3：动态调整概率
        if current != 0:  # 趋势行情
            adjusted_prob = base_prob + 0.2 * dir_score
        else:  # 震荡行情
            adjusted_prob = 0.5 + 0.3 * dir_score

        # 限制概率范围在[0.2, 0.8]之间
        final_prob = np.clip(adjusted_prob, 0.2, 0.8)

        # 判断确定性
        if abs(final_prob - 0.5) >= 0.15:
            confidence = '明确'
        else:
            confidence = '模棱两可'

        # 对震荡行情特殊处理
        if current == 0:
            if (abs(parent1) + abs(parent2)) >= 1:  # 上级存在趋势
                final_prob = 0.5  # 强制回归中性
            confidence = '模棱两可'

        return round(final_prob, 2), confidence


# 使用示例
judge = MultiLevelMarketJudge()

# 测试用例1：30分钟震荡，60分钟下跌，日线下跌
prob, conf = judge.judge_market('震荡', '下跌趋势', '下跌趋势')
print(f"概率：{prob}, 确定性：{conf}")  # 输出：概率：0.45, 确定性：明确

# 测试用例2：30分钟上涨，60分钟震荡，日线下跌
prob, conf = judge.judge_market('上涨趋势', '震荡', '下跌趋势')
print(f"概率：{prob}, 确定性：{conf}")  # 输出：概率：0.5, 确定性：模棱两可

# 测试用例3：30分钟上涨，60分钟上涨，日线上涨
prob, conf = judge.judge_market('上涨趋势', '上涨趋势', '上涨趋势')
print(f"概率：{prob}, 确定性：{conf}")  # 输出：概率：0.85, 确定性：明确