"""
免费的python服务器地址  https://www.pythonanywhere.com/user/zcode2318/
zcode2318
zhaot1993
Sunday 24 September 2023
我的web主页  http://zcode2318.pythonanywhere.com/

1、我选的flask web框架，它最轻，在file, mysite，flask_app.py里写代码

    主页为静态页面,前后端分离ajax 显示表格
    资产代码加跳转链接,点开任何一行,跳转到对应资产页，资产页由pyecharts实现
    对应资产页,展示当前代码日线买卖点,每笔买卖利润,tab标签显示多种策略
    每天下午三点以后,上传更新各个资产的日线html页面

2、然后在web reload，就能在  看见自己的网页

3、本地运行时，模板放文件夹里； 部署线上时，pyanywhere不能新建文件夹，所以指定template目录为当前目录，页面放当前目录，
同时删除 自己的包，自己的app，最下面的app.run  放开项目自带的注释
"""

from flask import Flask, render_template, request

# 项目自带的别动
# app = Flask(__name__, static_folder="./", template_folder='./')


# 以下是自己添加的
from RMQTool import Tools as RMTTools

# 本地环境 就本地回测时用
app = Flask(__name__, template_folder=RMTTools.read_config("RMQVisualized", "template_folder"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/barChart")  # 前后端分离用这个改
def get_bar_chart():
    # 收益率代码没写
    assets = {'000001': {'name': '上证指数', 'strategy1': 0.1, 'strategy2': 0, 'strategy3': 0, },
              }
    return assets


@app.route("/href")
def href_chart():
    code = request.args.get('code')
    code_list = ['000001', '510050', ]
    jump = False
    for i in code_list:
        if i == code:
            jump = True
    if jump:
        return render_template(code + ".html")
    else:
        return render_template("error.html")


"""
而pythonanywhere的flask封装好了这一步，不用run，直接reload就好了
相当于pythonanywhere执行了
    python flask_app.py  还在flask_app.py增加了 app.run()
"""
app.run()
