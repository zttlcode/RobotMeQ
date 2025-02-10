import pandas as pd
import numpy as np

# 创建示例 DataFrame，时间列作为索引
date_range = pd.date_range(start='2024-01-01', periods=10, freq='D')
df = pd.DataFrame({'A': np.random.randint(100, 200, size=10)}, index=date_range)

# 创建 C 列并设为 object 类型
df['C'] = None  # 先初始化列
df['C'] = df['C'].astype(object)  # 确保列类型可存储字典

# 遍历 DataFrame，为 C 列赋值字典
for index in df.index:
    df.at[index, 'C'] = {'date': index.strftime('%Y-%m-%d'), 'value': df.at[index, 'A']}

# 打印 DataFrame
print(df)
