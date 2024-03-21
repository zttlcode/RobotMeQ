import pandas as pd


class PositionEntity:
    def __init__(self):
        # 订单数据 二维的字典结构
        self.currentOrders = {}  # 记录买入订单 用于在策略里判断是否有仓位可以卖
        self.historyOrders = {}  # 记录已完成订单，卖出时更新，计算收益
        self.orderNumber = 0  # 每买一单+1
        self.money = 1000000  # 总资产100万
        self.trade_point_list = []  # 记录策略所有买卖点  格式 [["2021-04-26", 47, "buy"], ["2021-06-15", 55.1, "sell"]]


def buy(positionEntity, tradeDateTime, price, volume):
    positionEntity.orderNumber += 1  # 订单编号更新
    key = "order" + str(positionEntity.orderNumber)  # 准备key
    positionEntity.currentOrders[key] = {'openPrice': price, 'openDateTime': tradeDateTime, 'volume': volume}


def sell(positionEntity, tradeDateTime, key, price):
    # 给这个要卖的key，增加关闭价格、交易时间、计算本单收益：（卖价-买入价）*交易量-千分之一印花税-买卖两次的手续费万分之3
    positionEntity.currentOrders[key]['closePrice'] = price
    positionEntity.currentOrders[key]['closeDateTime'] = tradeDateTime
    positionEntity.currentOrders[key]['pnl'] = (price - positionEntity.currentOrders[key]['openPrice']) * \
                                               positionEntity.currentOrders[key]['volume'] - price * \
                                               positionEntity.currentOrders[key]['volume'] * 1 / 1000 - \
                                               (price + positionEntity.currentOrders[key]['openPrice']) * \
                                               positionEntity.currentOrders[key]['volume'] * 3 / 10000
    positionEntity.historyOrders[key] = positionEntity.currentOrders.pop(key)  # 把卖的订单，从当前仓位列表里，复制到历史仓位列表里


"""
python实现海龟策略，DataFrame 类型的参数 df，其中包含了四个价格数据列：low、high、open 和 close。
具体要求为：
1、分仓24个头寸，单个品种最多4个头寸，单个方向最多12个头寸；
2、入场策略，短线20天突破，长线55天突破。价格突破20天最高点，入场做多，若突破20天最低点，入场做空。55天同理。
3、止损点，入场前设置好止损点——atr指标中20天波动幅度的平均值设为n，止损点就是加或减2n，或0.5n，波动大用前者，波动小用后者。
4、出场：突破10天或20天的最高或最低位出场，比如多头，破10天最低，说明多头趋势被逆转，出场，空头同理。
5、加仓规则是价格在上次买入价格的基础上往盈利的方向变化（系数在 0.5～1 之间），即可在增加 25% 仓位。
6、海龟交易法同样具备两种止损规则:
    1 统一止损是任何一笔交易都不能出现账户规模 2% 以上的风险；
    2 双重止损是账户只承受 0.5%的账户风险，各单位头寸保持各自的止损点位不变。
    海龟交易法的卖出规则一旦出发都要退出。

"""
class TurtleTrader:

    def __init__(self, df):
        self.df = df
        self.positions = []
        self.unit_size = None
        self.multiplier = 2
        self.stop_loss_type = 'single'  # single是统一止损，dual是双重止损
        self.risk_per_trade = 0.02  # 1 统一止损是任何一笔交易都不能出现账户规模 2% 以上的风险；
        self.risk_per_unit = 0.005  # 2 双重止损是账户只承受 0.5%的账户风险，各单位头寸保持各自的止损点位不变。

    def set_unit_size(self, capital):
        self.unit_size = capital * 0.01 / 24 / self.risk_per_unit

    def set_stop_loss_type(self, stop_loss_type):
        self.stop_loss_type = stop_loss_type

    def set_risk_per_trade(self, risk_per_trade):
        self.risk_per_trade = risk_per_trade

    def set_risk_per_unit(self, risk_per_unit):
        self.risk_per_unit = risk_per_unit

    def get_unit_size(self):
        return self.unit_size

    def get_positions(self):
        return self.positions

    def calc_atr(self, n=20):
        tr = pd.DataFrame()
        tr['high-low'] = self.df['high'] - self.df['low']
        tr['high-pc'] = abs(self.df['high'] - self.df['close'].shift())
        tr['low-pc'] = abs(self.df['low'] - self.df['close'].shift())
        tr['tr'] = tr.max(axis=1)
        atr = tr['tr'].rolling(n).mean()
        return atr

    def enter_position(self, direction, price, atr):
        pos = {
            'direction': direction,
            'entry_price': price,
            'stop_loss': None,
            'units': 0,
            'multiplier': self.multiplier,
            'atr': atr
        }
        # 3、止损点，入场前设置好止损点——atr指标中20天波动幅度的平均值设为n，止损点就是加或减2n，或0.5n，波动大用前者，波动小用后者。
        pos['stop_loss'] = price - pos['direction'] * 2 * pos['atr']
        units = int(self.unit_size / (pos['atr'] * pos['multiplier']))
        units = min(units, 4 - len([p for p in self.positions if p['direction'] == direction]))
        units = min(units, 12 - len([p for p in self.positions]))
        pos['units'] = units
        self.positions.append(pos)

    def exit_position(self, position, price):
        direction = position['direction']
        units = position['units']
        self.positions.remove(position)
        for i in range(units):
            pnl = (price - position['entry_price']) * direction
            if pnl > 0:
                self.unit_size += self.risk_per_unit * self.unit_size
                position['multiplier'] = min(position['multiplier'] * 1.25, 2)
            else:
                self.unit_size -= self.risk_per_unit * self.unit_size
                position['multiplier'] = max(position['multiplier'] / 1.25, 1)

    def check_stop_loss(self, position, price):
        direction = position['direction']
        stop_loss = position['stop_loss']
        if direction * (price - stop_loss) >= 0:
            self.exit_position(position, stop_loss)

    def update_stop_loss(self, position, atr):
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = entry_price - direction * 2 * atr
        position['stop_loss'] = stop_loss

    def trade(self):
        atr = self.calc_atr()
        for i in range(55, len(self.df)):
            current_price = self.df['close'][i]

            prev_price = self.df['close'][i - 1]
            # 突破10天或20天的最高或最低位出场，这里用10天
            prev_lowest_low = self.df['low'][i - 10: i].min()
            prev_highest_high = self.df['high'][i - 10: i].max()

            if current_price > prev_highest_high:  # 突破55天最高点，入场做多
                if len([p for p in self.positions if p['direction'] == 1]) < 12:  # 单个方向最多12个头寸
                    self.enter_position(1, current_price, atr[i])

            if current_price < prev_lowest_low:  # 突破55天最低点，入场做空
                if len([p for p in self.positions if p['direction'] == -1]) < 12:
                    self.enter_position(-1, current_price, atr[i])

            for position in self.positions:
                direction = position['direction']
                if direction == 1:
                    exit_price = self.df['low'][i - 20: i].min()
                else:
                    exit_price = self.df['high'][i - 20: i].max()
                self.check_stop_loss(position, current_price)
                if direction * (current_price - exit_price) <= 0:
                    self.exit_position(position, exit_price)

            if self.stop_loss_type == 'dual':  # 默认是single统一止损，若双重止损，执行此函数
                for position in self.positions:
                    self.update_stop_loss(position, atr[i])

        # Close all remaining positions at the last available price
        # for position in self.positions:
        #     self.exit_position(position, self.df['close'].iloc[-1])

"""
使用上述代码，您可以创建一个TurtleTrader对象，并通过设置相关参数来执行海龟策略的回测。
在回测过程中，策略将根据给定的入场和出场条件进行交易，并根据止损规则管理头寸和风险。
最后，您可以使用get_positions方法获取最终的交易头寸。
"""

from RMQTool import Tools as RMTTools

if __name__ == '__main__':
    filePath = RMTTools.read_config("RMQData", "backTest_bar") + 'backtest_bar_000001_5.csv'
    df = pd.read_csv(filePath, encoding='gbk')
    # 1、创建海龟策略对象
    backtest_for_turtle = TurtleTrader(df)
    # 2、设置各种参数
    backtest_for_turtle.set_unit_size(10000000)  # 设置自己有1000万
    # 海龟交易法具备两种止损规则，默认是single统一止损，也可以设置dual双重止损
    # backtest_for_turtle.set_stop_loss_type('dual')
    # backtest_for_turtle.set_risk_per_unit() # 双重止损默认0.5% 这里可以修改参数

    # backtest_for_turtle.set_risk_per_trade() # 统一止损默认2% 这里可以修改参数

    # 3、策略执行
    backtest_for_turtle.trade()

    # 4、查看结果
    size = backtest_for_turtle.get_unit_size()
    print(size)
    position = backtest_for_turtle.get_positions()  # 获取最终的交易头寸
    print(position)