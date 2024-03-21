import statsmodels.formula.api as smf
import pandas as pd
import RMQStrategy.Indicator as RMQIndicator


# 最小二乘法拟合直线
def LeastSquares(df):
    # 参考博文如下
    # https://blog.csdn.net/sinat_23971513/article/details/121483023
    # https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/108687817
    # seaborn，也是个画图的包，官方文档地址：http://seaborn.pydata.org/generated/seaborn.lmplot.html
    # statsmodels，计算统计数据的包，官方文档地址：https://www.statsmodels.org/stable/index.html
    # scipy 也是个数学用的库

    # 这个暂定传df，但转为时间价格坐标系的代码还没写

    # 读取 时间、价格文件，返回斜率
    ccpp = pd.read_csv('D:\\workspace\\github\\RobotMeQuant\\QuantData\\ttt.csv')  # 第一列为x轴，第二列为y轴 y = ax + b
    # plt.show()  # 展示拟合图
    regression2 = smf.ols(formula='y~x', data=ccpp)  # y是被解释变量，x是解释变量 OLS Ordinary Least Squares 普通最小二乘法
    model2 = regression2.fit()
    print(model2.params)  # params是pandas.Series格式 Intercept是b,x是斜率a
    return model2.params.x


def calMACD_area(df):
    # 定义计算MACD红绿柱面积的方法
    # 拿到macd
    DataFrame = RMQIndicator.calMACD(df, 12, 26, 9)
    # 计算好macd面积区域存储在列表里
    macd_area_dic = []

    # 初始化临时变量
    macd_area_sum_temp_red = 0  # 临时累计上涨区域面积
    macd_area_sum_temp_green = 0  # 临时累计下跌区域面积

    # 每个区域的价格最值，先初始化
    highest = DataFrame.at[0, 'close']
    lowest = DataFrame.at[0, 'close']

    # 区域变更控制开关
    red_count = 0  # 进入红区累加，变更绿区时，重新归0，开启开关
    green_count = 0  # 进入绿区累加，变更红区时，重新归0，开启开关
    change = False  # 开关，只在变更区域时开启一次，其余时间为关闭状态

    # 遍历整个df
    for index, row in DataFrame.iterrows():
        if row['MACD'] > 0:  # 大于0则用红色
            if green_count > 0:  # 说明刚从绿区进来，开开关，重新归0
                change = True
                green_count = 0
                highest = row['high']  # 最高价初始化为当前区间第一个价格
            else:  # 说明不是刚进来，保持开关关闭
                change = False
            red_count += 1  # 说明此时是红区状态，一旦下次进去绿区，会被归0
            macd_area_sum_temp_red += row['MACD']  # 红色面积累加
            highest = max(highest, row['high'])  # 记录红区最高价
        elif row['MACD'] < 0:  # 小于0则用绿色
            if red_count > 0:  # 说明刚从红区进来，开开关，重新归0
                change = True
                red_count = 0
                lowest = row['low']  # 最低价初始化为当前区间第一个价格
            else:  # 说明不是刚进来，保持开关关闭
                change = False
            green_count += 1  # 说明此时是绿区状态，一旦下次进去红区，会被归0
            macd_area_sum_temp_green += row['MACD']
            lowest = min(lowest, row['low'])

        # 否则是空文件，都不处理

        if change:
            if red_count == 0:
                # 进入新区域，要把前一个区域的结束时间填上，index代表当前区域，index-1是前一个区域的下标
                macd_area_dic.insert(0, {'area': macd_area_sum_temp_red, 'price': highest,
                                         'time': DataFrame.at[index - 1, 'time']})
                macd_area_sum_temp_red = 0
                highest = 0
            else:
                macd_area_dic.insert(0, {'area': macd_area_sum_temp_green, 'price': lowest,
                                         'time': DataFrame.at[index - 1, 'time']})
                macd_area_sum_temp_green = 0
                lowest = 0

    # 循环结束，判断计算最后一个区域的面积，最后一个就是当前区域
    if red_count > 0:
        # 最后一个是红区
        macd_area_dic.insert(0, {'area': macd_area_sum_temp_red, 'price': highest,
                                 'time': DataFrame.at[len(DataFrame) - 1, 'time']})
    elif green_count > 0:
        # 最后一个是绿区
        macd_area_dic.insert(0, {'area': macd_area_sum_temp_green, 'price': lowest,
                                 'time': DataFrame.at[len(DataFrame) - 1, 'time']})
    # 否则是空文件，都不处理
    result_DataFrame = pd.DataFrame(macd_area_dic)
    # result_DataFrame共2列，面积为正，对应最高价，反之最低价。0是最新数据
    return result_DataFrame
