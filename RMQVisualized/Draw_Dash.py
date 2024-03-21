"""
flask整合dash
pythonanywhere 新建了flask项目，怎么让plotly的dash表格放上去呢  步骤如下
目前是主页html写好表格，ajax直接跳转图表
如果以后要动态生成页面互动表格，用dash
"""

from flask import Flask

# 以下是自己添加的
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

# 项目自带的别动
app = Flask(__name__)
# 自己new一个Dash对象，把Flask的对象给server，得到了一个Dash的Flask对象
appDash = Dash(__name__, server=app)


# -------------------------------下面写自己的代码-------------------------
def plot_cand_volume(data):
    # https://zhuanlan.zhihu.com/p/469985462?utm_id=0
    # 删除不交易时间
    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('', '成交量'),
                        row_width=[0.2, 0.7])

    # 绘制k数据
    fig.add_trace(go.Candlestick(x=data["time"], open=data["open"], high=data["high"],
                                 low=data["low"], close=data["close"], name=""),
                  row=1, col=1
                  )

    # 绘制成交量数据
    fig.add_trace(go.Bar(x=data['time'], y=data['volume'], showlegend=False), row=2, col=1)

    fig.update_xaxes(
        title_text='time',
        rangeslider_visible=True,  # 下方滑动条缩放
        rangeselector=dict(
            # 增加固定范围选择
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(step='all')])))

    # Do not show OHLC's rangeslider plot
    fig.update(layout_xaxis_rangeslider_visible=False)

    return fig


# 然后 appDash.layout = html.Div([ 这里填自己的代码就行了  ])
appDash.layout = html.Div([
    html.H4('Interactive color selection with simple Dash example'),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.P("Select color:"),
    dcc.Dropdown(
        id="dropdown",
        options=['Red', 'Blue', 'Green'],
        value='Red',
        clearable=False,
    ),
    dcc.Graph(id="graph"),
])


@appDash.callback(
    Output("graph", "figure"),  # 输出是给id为graph的一个figure对象
    Input("dropdown", "value"))  # 输入是取id为dropdown的value属性
def display_pic(color):
    fig = go.Figure(
        data=go.Bar(y=[2, 3, 1],  # replace with your own data source
                    marker_color=color))

    # 只要return figure对象就行  这里可以写自己代码
    # filePath = 'E:\\PycharmProjects\\QuantData\\backTest\\backtest_bar_601012_30.csv'
    # data = pd.read_csv(filePath, encoding='gbk')
    # fig = plot_cand_volume(data)
    return fig


# 项目自带的 暂时用不到
# @app.route('/')
# def hello_world():
#     return render_template("index3.html")
# -------------------------------上面是写自己的代码-------------------------


"""
而pythonanywhere的flask封装好了这一步，不用run，直接reload就好了
相当于pythonanywhere执行了
    python flask_app.py  还在flask_app.py增加了 app.run_server(debug=True)
所以我把flask_app.py代码复制到本地，只要增加一个appDash.run_server(debug=True)  也能有和pythonanywhere一样的效果
"""
appDash.run_server(debug=True)  # 如果是本地运行，就直接  appDash.run_server(debug=True)

