/RMQData  数据处理
    Asset.py 4种资产：A股的股票、指数、ETF基金，数字币。asset对象包含bar转化、指标计算、仓位记录 三个属性。
    Bar.py  只有A股会用到，主要用于实盘时把tick转为bar，回测时为了模拟实盘又将bar转为tick去回测，
            之所以这么做是因为当初我认为每秒的tick都要当作收盘价，这样策略反应最快。但缺点是bar没结束导致最终close不一定满足策略，后来改为bar结束再进策略
    HistoryData.py  从证券宝获取A股股票全级别数据、A股指数日线级别数据
                    从通达信获取A股指数分钟级别数据
                    从akshare获取美股标普500成分股日线数据、港股前100及任意900股票的日线级别数据
    HistoryData_crypto.py 记录了使用币安官方库下载历史数据，再处理为我需要的格式
    Indicator.py  这是Asset.py中asset对象的指标计算，主要用于在策略中计算指标。
                    还为tea策略增加了 多级别交流指标值、我发明的背离计算、记录本级别背离了几次。为这个策略付出太多了，白费
    Position.py   这是Asset.py中asset对象的仓位记录，只有全仓买卖的记录，没有仓位管理方法，也没必要
                    我增加了个移动止损，当买入后价格从最高位回落2%，就会触发偷偷止损，但仓位没卖。A股实践中止损太频繁了
    Tick.py  A股专用获取实盘tick，并把tick给Bar.py转为bar数据
/RMQModel  为训练模型处理数据
    Dataset.py  把标注过分类的数据转为ts
    Evaluate.py  计算收益率
    Label.py  把回测数据标注了
/RMQStrategy  策略
/RMQTool  工具箱
    Message.py  实盘时发邮件、发各种app消息
    Tdx_auto.py  用自动化去通达信客户端下载ETF各级别数据，用于实盘
    Tools.py  读写配置文件，获取新得一年的交易日
    Tools_model.py  RMQModel专用的工具箱
    Tools_time_seris_forecast.py  当初时序模型回归的工具箱
/RMQVisualized  可视化
    Deploy_Flask.py  本地部署前端页面，也是pythonanywhere的主页
    Draw_Dash.py  Dash是在python代码里写前端代码，对我没啥用了
    Draw_Pyecharts.py  用来画交易点的

Run.py是跑回测的
使用Run类的py运行策略，有新增策略时，在RMQStrategy/Strategy.py中新增， 同时在config.ini和config_prd.ini中增加新策略 的回测点保存路径



