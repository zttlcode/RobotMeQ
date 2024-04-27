import pandas as pd
import json
from RMQTool import Tools as RMTTools


class PositionEntity:
    def __init__(self, indicatorEntity):
        # 订单数据 二维的字典结构
        self.currentOrders = {}  # 记录买入订单 用于在策略里判断是否有仓位可以卖
        self.historyOrders = {}  # 记录已完成订单，卖出时更新，计算收益
        self.orderNumber = 0  # 每买一单+1
        self.money = 1000000  # 总资产100万
        self.trade_point_list = []  # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]

        # 尝试读取JSON文件
        try:
            with open(RMTTools.read_config("RMQData", "position_currentOrders")
                      + "position_"
                      + indicatorEntity.IE_assetsCode
                      + "_"
                      + indicatorEntity.IE_timeLevel
                      + ".json", 'r') as file:
                self.currentOrders = json.load(file)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass


def buy(positionEntity, indicatorEntity, price, volume):
    positionEntity.orderNumber += 1  # 订单编号更新
    key = "order" + str(positionEntity.orderNumber)  # 准备key
    positionEntity.currentOrders[key] = {'openPrice': price,
                                         'openDateTime': indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S'),
                                         'volume': volume}
    # 将仓位信息保存到文件
    with open(RMTTools.read_config("RMQData", "position_currentOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'a') as file:
        json.dump(positionEntity.currentOrders, file)


def sell(positionEntity, indicatorEntity, key, price):
    # 给这个要卖的key，增加关闭价格、交易时间
    positionEntity.currentOrders[key]['closePrice'] = price
    positionEntity.currentOrders[key]['closeDateTime'] = indicatorEntity.tick_time.strftime('%Y-%m-%d %H:%M:%S')
    # 计算本单收益 =（卖价-买入价）* 交易量 - 千分之一印花税 - 买卖两次的手续费万分之3
    positionEntity.currentOrders[key]['pnl'] = ((price - positionEntity.currentOrders[key]['openPrice'])
                                                * positionEntity.currentOrders[key]['volume']
                                                - price * positionEntity.currentOrders[key]['volume'] * 1 / 1000
                                                - (price + positionEntity.currentOrders[key]['openPrice'])
                                                * positionEntity.currentOrders[key]['volume'] * 3 / 10000)
    positionEntity.historyOrders[key] = positionEntity.currentOrders.pop(key)  # 把卖的订单，从当前仓位列表里，复制到历史仓位列表里

    # 将交易记录保存到文件
    with open(RMTTools.read_config("RMQData", "position_historyOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'a') as file:
        json.dump(positionEntity.historyOrders, file)

    # 清空仓位信息
    with open(RMTTools.read_config("RMQData", "position_currentOrders")
              + "position_"
              + indicatorEntity.IE_assetsCode
              + "_"
              + indicatorEntity.IE_timeLevel
              + ".json", 'w') as file:
        json.dump({}, file)
