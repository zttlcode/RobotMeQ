from typing import List, Sequence, Union
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Kline, Line, Bar, Grid, Tab
import pandas as pd

from RMQTool import Tools as RMTTools
import RMQData.Indicator as RMQIndicator
import RMQData.Asset as RMQAsset


def split_data(origin_data) -> dict:
    datas = []
    times = []
    vols = []
    difs = []
    deas = []
    macds = []

    for row in origin_data.index:
        """
            接收的Dataframe列顺序如下  time下标为0
            time, open, close, low, high, volume, EMA, DIF, DEA, MACD
            origin_data.loc[row][1:5]表示dataframe的第row行的1~4列
            得到一个series对象，series对象是个特殊的字典，是键值对形式，而这里需要只有value的list，所以加.values，
            得到ndarry对象，再转list
        """
        datas.append(list(origin_data.loc[row][1:5].values))
        times.append(origin_data.loc[row][0:1][0])
        vols.append(origin_data.loc[row][5])
        difs.append(origin_data.loc[row][7])
        deas.append(origin_data.loc[row][8])
        macds.append(origin_data.loc[row][9])

    vols = [int(v) for v in vols]

    return {
        "datas": datas,
        "times": times,
        "vols": vols,
        "difs": difs,
        "deas": deas,
        "macds": macds,
    }


def split_data_part(data, trade_point_list) -> Sequence:
    """
    买卖点显示到图表上  格式如下
    trade_point_list [["2022-01-01","18.25","buy"],["2022-01-05","20.25","sell"]]
    [:10]
    """
    mark_line_data = []
    for i in range(len(data["times"])):  # 遍历x轴所有时间
        if trade_point_list:  # 列表不空才比较，如果空，直接退出循环，后面的不用比了
            # 如果展示分钟级别，则改成 trade_point_list[0][0][:10] == data["times"][i][:10]，同一天就留下
            # 展示单级别分钟的，把[:10]去掉
            # nature_quant过滤交易点，则trade_point_list[0][0]精确到日，也把[:10]去掉
            if trade_point_list[0][0] == data["times"][i]:  # 时间对上了
                mark_line_data.append(
                    [
                        {
                            "xAxis": i,  # 找到对应的x轴的值
                            "yAxis": float("%.3f" % trade_point_list[0][1]),  # y就是买卖价格
                            # "value": trade_point_list[0][2] + ":" + str(trade_point_list[0][1]),  # 显示买还是卖
                            "value": trade_point_list[0][2][0:1],  # 显示买还是卖
                        },
                        {
                            "xAxis": i,  # 找到对应的x轴的值
                            "yAxis": float("%.3f" % trade_point_list[0][1]),  # y就是买卖价格
                        },
                    ]
                )
                del trade_point_list[0]  # 对上一个删一个，这样就不用双重循环了，永远只比较第一个
        else:
            break
    return mark_line_data


def calculate_ma(data, day_count: int):
    result: List[Union[float, str]] = []

    for i in range(len(data["times"])):
        if i < day_count:
            result.append("-")
            continue
        sum_total = 0.0
        for j in range(day_count):
            sum_total += float(data["datas"][i - j][1])
        result.append(abs(float("%.2f" % (sum_total / day_count))))
    return result


def draw_chart(data, trade_point_list):
    # 这个管的就是K线
    kline = (
        Kline()
        .add_xaxis(xaxis_data=data["times"])
        .add_yaxis(
            series_name="",
            y_axis=data["datas"],
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",  # 红色
                color0="#14b143",  # 绿色
                border_color="#ef232a",
                border_color0="#14b143",
            ),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"),
                    opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            # 这个原本管的是两个交易日之间，每天累积的成交量之和 比如今天成交量55M，10天后的成交量78M，中间这10天成交量一共511
            # 资产只用日线级别, 改为标记买卖点及买卖价格
            markline_opts=opts.MarkLineOpts(
                label_opts=opts.LabelOpts(
                    position="middle", color="blue", font_size=15
                ),
                data=split_data_part(data, trade_point_list),
                symbol=["circle", "none"],
            ),
        )
        # 这个估计是箱体区域,就是均线回归策略中,上下震荡的区间,我猜的
        # .set_series_opts(
        #     markarea_opts=opts.MarkAreaOpts(is_silent=True, data=split_data_part(data, trade_point_list))
        # )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="K线周期图表", pos_left="0"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False, type_="inside", xaxis_index=[0, 0], range_end=100
                ),
                opts.DataZoomOpts(
                    is_show=True, xaxis_index=[0, 1], pos_top="97%", range_end=100
                ),
                opts.DataZoomOpts(is_show=False, xaxis_index=[0, 2], range_end=100),
            ],
        )
    )
    # 这个管的是均线
    kline_line = (
        Line()
        .add_xaxis(xaxis_data=data["times"])
        .add_yaxis(
            series_name="MA5",
            y_axis=calculate_ma(data, day_count=5),
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="MA10",
            y_axis=calculate_ma(data, day_count=10),
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                split_number=3,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
        )
    )
    # Overlap Kline + Line
    # overlap意思是层叠多图 把蜡烛图和均线组合在一起了  啥原理呢?无论多少个图,横坐标一样就行,它会自动组合
    overlap_kline_line = kline.overlap(kline_line)

    # Bar-1
    # 这个是成交量的柱状图
    bar_1 = (
        Bar()
        .add_xaxis(xaxis_data=data["times"])  # 横坐标还是跟上面的一致
        .add_yaxis(
            series_name="Volumn",
            y_axis=data["vols"],
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            # 改进后在 grid 中 add_js_funcs 后变成如下  下面不是注释，是JS代码
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                function(params) {
                    var colorList;
                    if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                        colorList = '#ef232a';
                    } else {
                        colorList = '#14b143';
                    }
                    return colorList;
                }
                """
                )
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    # Bar-2 (Overlap Bar + Line)
    # 这个是macd的柱状图 下面line_2是macd的折线图
    bar_2 = (
        Bar()
        .add_xaxis(xaxis_data=data["times"])
        .add_yaxis(
            series_name="MACD",  # 这是每个柱子的名字,鼠标放上去就会显示 MACD:多少
            y_axis=data["macds"],
            xaxis_index=2,  # 使用的 x 轴的 index，在单个图表实例中存在多个 x 轴的时候有用。
            yaxis_index=2,  # 使用的 y 轴的 index，在单个图表实例中存在多个 y 轴的时候有用。 没明白是干啥的
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                        function(params) {
                            var colorList;
                            if (params.data >= 0) {
                              colorList = '#ef232a';
                            } else {
                              colorList = '#14b143';
                            }
                            return colorList;
                        }
                        """
                )
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=2,
                split_number=4,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    # 两根线,所以有两个add_yaxis  跟上面的均线一样的
    line_2 = (
        Line()
        .add_xaxis(xaxis_data=data["times"])
        .add_yaxis(
            series_name="DIF",
            y_axis=data["difs"],
            xaxis_index=2,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="DEA",
            y_axis=data["deas"],
            xaxis_index=2,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
    )
    # 最下面的柱状图和折线图
    overlap_bar_line = bar_2.overlap(line_2)

    # 最后的 Grid
    grid_chart = Grid()

    # 这个是为了把 data.datas 这个数据写入到 html 中,还没想到怎么跨 series 传值
    # demo 中的代码也是用全局变量传的
    grid_chart.add_js_funcs("var barData = {}".format(data["datas"]))

    # K线图和 MA 的折线图
    grid_chart.add(
        overlap_kline_line,
        grid_opts=opts.GridOpts(pos_left="3%", pos_right="1%", height="60%"),
    )
    # Volumn 柱状图
    grid_chart.add(
        bar_1,
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
        ),
    )
    # MACD DIFS DEAS
    grid_chart.add(
        overlap_bar_line,
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="82%", height="14%"
        ),
    )
    return grid_chart


def draw_pyecharts(dataFrame, tplList, code):
    # 1、计算macd
    windowDF = RMQIndicator.calMACD(dataFrame)
    # 2、交换列顺序：开高低收 变为 开收低高，其他没变——原始顺序 time, open, high, low, close, volume, EMA, DIF, DEA, MACD
    windowDF = pd.DataFrame(windowDF, columns=['time', 'open', 'close', 'low', 'high', 'volume', 'EMA', 'DIF', 'DEA', 'MACD'])
    # 3、保留两位小数
    windowDF = windowDF.round(2)

    # 4、df转为字典
    data = split_data(origin_data=windowDF)

    """
      5、展示结果
        1）多个策略生成tab
        2）单个策略生成单页
    """
    if len(tplList) > 1:
        # 1）多表格  等策略多了，就用下面的tab  增加策略,就继续加tab.add(……)
        tab = Tab()
        for tpl in tplList:
            tab.add(draw_chart(data, tpl[0]), tpl[1])
        tab.render(RMTTools.read_config("RMQVisualized", "template_folder") + code + ".html")
    else:
        # 2）单表格
        grid_chart = draw_chart(data, tplList[0][0])
        grid_chart.width = "98%"
        grid_chart.height = "700px"
        grid_chart.render(RMTTools.read_config("RMQVisualized", "template_folder") + code + ".html")


if __name__ == "__main__":
    """
        python可视化画蜡烛图 https://gallery.pyecharts.org/#/Candlestick/professional_kline_chart
        官网 https://pyecharts.org/#/
        pyecharts整合Flask  https://pyecharts.org/#/zh-cn/web_flask
        https://echarts.apache.org/examples/zh/index.html#chart-type-candlestick
    """
    assetList = RMQAsset.asset_generator('000032', '', ['5', '15', '30', '60', 'd'], 'stock', 1)
    # assetList = RMQAsset.asset_generator('BTCUSDT', 'BTC', ['15', '60', '240', 'd'], 'crypto')

    # 读取日线数据
    filePath = RMTTools.read_config("RMQData", "backtest_bar") + 'backtest_bar_' + assetList[0].assetsCode + '_d.csv'
    df = pd.read_csv(filePath, encoding='utf-8')

    # # 图表派策略买卖点
    # tpl_filepath = RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_"
    # df_tpl_5 = pd.read_csv(tpl_filepath + assetList[0].assetsCode + "_" + assetList[0].barEntity.timeLevel + ".csv")
    # df_tpl_15 = pd.read_csv(tpl_filepath + assetList[1].assetsCode + "_" + assetList[1].barEntity.timeLevel + ".csv")
    # df_tpl_30 = pd.read_csv(tpl_filepath + assetList[2].assetsCode + "_" + assetList[2].barEntity.timeLevel + ".csv")
    # df_tpl_60 = pd.read_csv(tpl_filepath + assetList[3].assetsCode + "_" + assetList[3].barEntity.timeLevel + ".csv")
    # df_tpl_d = pd.read_csv(tpl_filepath + assetList[4].assetsCode + "_" + assetList[4].barEntity.timeLevel + ".csv")
    # # 各级别买卖点合并成一个df
    # df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60, df_tpl_d])
    # # df_tpl = pd.concat([df_tpl_5, df_tpl_15, df_tpl_30, df_tpl_60])
    # df_tpl.sort_values(by="0", axis=0, inplace=True)  # 按第一列（日期）排序,"0"是列名，axis=0表示按列，axis=1按行，在原数据上修改
    # # 后面split_data_part函数画交易点位时，是对比交易点日期和 df的日期，同一天几个信号只会比较一次，因此交易点只保留第一个日期。下面是去重
    # # 将时间列解析为日期格式
    # df_tpl['date_only'] = pd.to_datetime(df_tpl['0']).dt.date  # 提取日期部分
    # # 按日期去重，保留重复的第一行
    # df_tpl = df_tpl.drop_duplicates(subset='date_only', keep='first')
    # # 删除辅助列 'date_only'（如果不需要保留）
    # df_tpl = df_tpl.drop(columns=['date_only'])
    # trade_point_list_tbp = df_tpl.values.tolist()  # df转列表

    # 使用nature_quant过滤交易点后，再次可视化交易点位
    df_labeled = pd.read_csv((RMTTools.read_config("RMQData", "trade_point_backtest") + "trade_point_list_" +
     assetList[0].assetsCode + "_concat_labeled" + ".csv"), encoding='utf-8', parse_dates=["time"])
    # 过滤出 label 为 1 或 3 的行
    filtered_df = df_labeled[df_labeled["label"].isin([1, 3])]
    # 去掉 time 列的时分秒，只保留日期
    filtered_df["time"] = filtered_df["time"].dt.date
    # 按 time 去重，只保留同一天的第一行数据
    unique_df = filtered_df.drop_duplicates(subset=["time"], keep="first")
    unique_df['time'] = unique_df['time'].astype(str)
    trade_point_list_tbp = unique_df.values.tolist()  # df转列表

    # trade_point_list_tbp = []
    # trade_point_list_tbp = [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]
    # trade_point_list_hg = [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]  # 海龟策略买卖点
    # trade_point_list_jx = [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]  # 均线策略买卖点

    # 组成列表集合
    tpl_list = [[trade_point_list_tbp, assetList[0].assetsCode+"-图表派策略"],
                # [trade_point_list_hg, assetCode + "-海龟策略"],
                # [trade_point_list_jx, assetCode + "-均线策略"],
                ]

    # 生成图表
    draw_pyecharts(df, tpl_list, assetList[0].assetsCode)
    # 本地运行，每天收盘后，生成html页面，上传到pythonanywhere

