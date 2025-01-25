import pandas as pd

from RMQTool import Tools as RMTTools
import RMQData.Asset as RMQAsset


def cal_return_rate(assetList, flag, strategy_name):
    # 加载数据
    item = 'trade_point_backtest_' + strategy_name
    df_filePath = (RMTTools.read_config("RMQData", item)
                   + assetList[0].assetsMarket
                   + "_"
                   + assetList[0].assetsCode + str(flag) + ".csv")

    # 读取CSV文件
    df = pd.read_csv(df_filePath)
    # 如果有 4 列，是标注后数据，过滤有效交易点
    if df.shape[1] == 4:
        dataframe = df[df['label'].isin([1, 3])].drop(columns=['label'])
    else:
        dataframe = df

    # 初始化资金和持仓状态
    holding_value = 0  # 市值
    shares = 0  # 持仓
    cost_per_share = 0  # 每股成本
    previous_total_cost = 0  # 总投资额
    previous_return_rate = 0.0  # 收益率

    latest_total_cost = 0  # 进行当日操作后的总投资额
    latest_return_rate = 0.0  # 进行当日操作后的收益率

    # 遍历CSV文件数据
    for index, row in dataframe.iterrows():
        price = row['price']
        signal = row['signal']

        # 新价格出现，更新市值
        holding_value = shares * price

        # 计算目前收益率
        if previous_total_cost != 0:
            profit_or_loss = holding_value - previous_total_cost  # 当前盈利或亏损金额= 市值-总投资额
            previous_return_rate = profit_or_loss / previous_total_cost  # 之前收益率= 盈亏/总投资额

        # 根据信号调整持仓和总花费
        if signal == 'buy':
            latest_total_cost = previous_total_cost + 100 * price  # 总投资额增加
            shares += 100  # 增加持股数
        elif signal == 'sell' and shares >= 100:
            latest_total_cost = previous_total_cost - 100 * price  # 总投资额减少
            shares -= 100  # 减少持股数
        else:
            latest_total_cost = previous_total_cost  # 无操作时总花费不变

        # 计算最新收益率
        holding_value = shares * price  # 更新持股金额
        if latest_total_cost != 0:
            profit_or_loss = holding_value - latest_total_cost  # 最新盈利或亏损金额
            latest_return_rate = profit_or_loss / latest_total_cost  # 最新收益率
            cost_per_share = latest_total_cost / shares if shares != 0 else 0  # 每股成本
        else:
            latest_return_rate = 0.0  # 防止除以零

        # 打印当前状态
        # print(f"时间: {row['time']}, 价格: {price}, 总成本: {previous_total_cost:.2f}, 收益率: {previous_return_rate:.2%}")
        # print(f"{signal} 100股, 目前持股数: {shares}, 持股金额: {holding_value:.2f}")
        # print(f"总成本: {latest_total_cost:.2f}, 收益率: {latest_return_rate:.2%}\n")

        # print(f"时间: {row['time']}, 现价: {price}, 总投资额: {previous_total_cost:.2f}, "
        #       f"收益率: {previous_return_rate:.2%}")
        # print(f"{signal} 100股, 持仓: {shares}, 市值: {holding_value:.2f}, 现价: {price}, "
        #       f"成本: {cost_per_share:.2f}, 收益率变为: {latest_return_rate:.2%}")
        # print(f"总投资额变为: {latest_total_cost:.2f}\n")

        # 更新之前总花费
        previous_total_cost = latest_total_cost

    # 遍历完成后，计算最终状态
    holding_value = shares * df.iloc[-1]['price']  # 最后一个价格计算持股价值
    if latest_total_cost != 0:
        final_profit_or_loss = holding_value - latest_total_cost
        final_return_rate = final_profit_or_loss / latest_total_cost
    else:
        final_return_rate = 0.0

    print(f"{assetList[0].assetsCode}{flag} 最终结果 持股数: {shares}, 市值: {holding_value:.2f}, "
          f"总投资额: {latest_total_cost:.2f}, 持股收益率: {final_return_rate:.2%}")

    # return round(final_return_rate, 4)


def compare_return_rate():
    # 计算不同标注方式的收益率
    allStockCode = pd.read_csv("./QuantData/a800_stocks.csv")
    n = 0
    temp_data_dict = {'have5label': []}
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['5', '15', '30', '60', 'd'],
                                             'stock',
                                             1, 'A')

        # 计算收益率  _5 _15 _30 _60 _d _concat _concat_labeled
        # have5 = cal_return_rate(assetList, "_concat_labeled", "tea_radical_nature")
        # no5 = cal_return_rate(assetList, "_concat")
        # temp_data_dict['have5label'].append(have5)
        # temp_data_dict['no5'].append(no5)
    print(n)
    result_df = pd.DataFrame(temp_data_dict)
    result_df.to_csv("./QuantData/temp.csv", index=False)
