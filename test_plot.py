import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 读取CSV文件
df = pd.read_csv('D:/github/dataset/backtest_bar_600332_30.csv', index_col='date', parse_dates=True)

# 截取最后100行数据的'close'列
df_last_100 = df['OT'].tail(30)

# 绘制实际值的曲线图
plt.figure(figsize=(20, 6))
plt.plot(df_last_100.index, df_last_100.values, color='red', label='Actual Close', linewidth=0.5)

# 加载预测数据
# data = np.load('D:/github/PatchTST/PatchTST_supervised/results/test_PatchTST_custom_ftMS_sl192_ll96_pl48_dm500_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
# data = np.load('D:/github/Autoformer/results/test_Autoformer_custom_ftMS_sl192_ll96_pl48_dm512_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
# data = np.load('D:/github/LTSF-Linear/results/test_DLinear_custom_ftMS_sl16_ll8_pl4_dm512_nh20_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')
data = np.load('D:/github/PatchTST/PatchTST_supervised/results/test_PatchTST_custom_ftMS_sl96_ll48_pl24_dm500_nh20_el2_dl1_df32_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy')

# 使用切片保留后两维
data_sliced = data[0, :, :]
close_column = data_sliced[:, -1]
print(close_column)
close_column_reshaped = close_column.reshape(-1, 1)
arr_raveled = close_column_reshaped.ravel()
# 绘制预测值的曲线图
plt.plot(df_last_100.index[-24:], arr_raveled, color='blue', label='Predicted Close')

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Close Values Over Time')
plt.xlabel('Time')
plt.ylabel('Close Value')

# 显示图形
plt.show()
