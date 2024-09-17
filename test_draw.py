import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 读取CSV文件
df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

# 截取最后100行数据的'close'列
df_Y = df['OT'].tail(500)

# 绘制实际值的曲线图
plt.figure(figsize=(10, 6))
plt.plot(df_Y.index, df_Y.values, color='red', label='Actual Close', linewidth=0.5)

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Close Values Over Time')
plt.xlabel('Time')
plt.ylabel('Close Value')

# 显示图形
plt.show()
