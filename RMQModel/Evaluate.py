import pandas as pd
import os
from RMQTool import Tools as RMTTools
import RMQData.Asset as RMQAsset


def cal_return_rate(asset, flag, strategy_name, pred, pred_tpp, handled_uneven):
    # 加载数据
    item = 'trade_point_backtest_' + strategy_name
    df_filePath = (RMTTools.read_config("RMQData", item)
                   + asset.assetsMarket
                   + "_"
                   + asset.assetsCode
                   + str(flag) + ".csv")
    if not os.path.exists(df_filePath):
        return None

    if handled_uneven:
        handled_uneven_filepath = (RMTTools.read_config("RMQData", item)
                                   + asset.assetsMarket
                                   + "_"
                                   + asset.assetsCode
                                   + str(flag)
                                   + "_handled_uneven" + ".csv")
        if not os.path.exists(handled_uneven_filepath):
            print(asset.assetsCode + "样本不均处理文件不存在")
            return None
        df_filePath = handled_uneven_filepath

    # 读取CSV文件
    df = pd.read_csv(df_filePath)
    df = df[df['time'] > '2019-01-01']
    if pred:
        if pred_tpp:
            df_prd_true_filePath = "D:/github/Time-Series-Library-Quant/results/" + asset.assetsCode + "_prd_result_tpp.csv"
            if not os.path.exists(df_prd_true_filePath):
                print(asset.assetsCode + "预测结果文件不存在")
                return None
            df_prd_true = pd.read_csv(df_prd_true_filePath)
            df['label'] = df_prd_true['predictions']
            df['predictions_market'] = df_prd_true['predictions_market']
            df = df[df['label'].isin([1, 3])]
            df = df[df["label"] == df["predictions_market"]]
            df = df.drop(columns=['predictions_market'])
        else:
            df_prd_true_filePath = "D:/github/Time-Series-Library-Quant/results/" + asset.assetsCode + "_prd_result.csv"
            if not os.path.exists(df_prd_true_filePath):
                print(asset.assetsCode + "预测结果文件不存在")
                return None
            df_prd_true = pd.read_csv(df_prd_true_filePath)
            df['label'] = df_prd_true['predictions']
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
        #
        # print(f"时间: {row['time']}, 现价: {price}, 总投资额: {previous_total_cost:.2f}, "
        #       f"收益率: {previous_return_rate:.2%}")
        # print(f"{signal} 100股, 持仓: {shares}, 市值: {holding_value:.2f}, 现价: {price}, "
        #       f"成本: {cost_per_share:.2f}, 收益率变为: {latest_return_rate:.2%}")
        # print(f"总投资额变为: {latest_total_cost:.2f}\n")

        # 更新之前总花费
        previous_total_cost = latest_total_cost

    # 遍历完成后，计算最终状态
    try:
        holding_value = shares * df.iloc[-1]['price']  # 最后一个价格计算持股价值
        if latest_total_cost != 0:
            final_profit_or_loss = holding_value - latest_total_cost
            #
            #
            """
             2025 05 06 做个修改，判断金额会让交易频繁的占优势，但我的过滤本就对交易次数做了减少
             因此还是改为交易率吧，只是收益上限变成了100%，下限变成了-100%，当然很多股票的上下限不止这点，但没有
             投入金额做参照，东方财富也是这么统计收益率的。
             我知道我的交易点如果按照收益率止盈，一定比传统赚的多，但止盈标准不好确定，移动止盈？固定金额？趋势策略是会让
             利润奔跑的，又要做很多实验，算了。
             就用收益率吧，有没有效果也就这样了。
            """
            if latest_total_cost < 0:
                final_return_rate = final_profit_or_loss / (-latest_total_cost)
            else:
                final_return_rate = final_profit_or_loss / latest_total_cost

            # final_return_rate = final_profit_or_loss / latest_total_cost
        else:
            final_return_rate = 0.0


        # print(f"{asset.assetsCode}{flag} 最终结果 持股数: {shares}, 市值: {holding_value:.2f}, "
        #       f"总投资额: {latest_total_cost:.2f}, 持股收益率: {final_return_rate:.2%}, "
        #       f"总收益: {(holding_value-latest_total_cost):.2f}")
        """
        返回收益率，会导致空仓后，赚钱赔钱的收益都是-100%，原因如下
        假如赚钱了，总投资额是-1000，仓位是0空仓，收益是 仓位-总投资额是 0-（-1000）是1000
        收益率= 1000 / （-1000）是-100%
        假如赔钱了，总投资额是 1000，仓位是0空仓，收益是 仓位-总投资额是 0-（ 1000）是-1000
        收益率= -1000 / （1000）是-100%
        赚钱的改成正100%可以，但有可能人家上限不是100%，得拿个参照物，用股价参照？没法界定翻了多少倍，这取决于持仓多少股，
        所以，都是一样的参照不如都不参照，直接按收益金额来，只要不同策略对比最终收益金额就好
        不同策略的股票是一样的，因此可以对比
        """
        return round(final_return_rate, 4)  # 2025 05 06 改回收益率，改了赚钱收益率-100%的bug
        # return round(holding_value-latest_total_cost, 4)
    except Exception as e:
        print(e, asset.assetsCode)


def return_rate(assetList, is_concat, flag, strategy_name, pred, pred_tpp, handled_uneven):
    res = None
    if is_concat:
        res = cal_return_rate(assetList[0], flag, strategy_name, pred, pred_tpp, handled_uneven)
    else:
        for asset in assetList:
            if flag:  # flag不是None
                res = cal_return_rate(asset, "_" + asset.barEntity.timeLevel + str(flag), strategy_name, pred, pred_tpp, handled_uneven)
            else:
                res = cal_return_rate(asset, "_" + asset.barEntity.timeLevel, strategy_name, pred, pred_tpp, handled_uneven)

    # print(assetList[0].assetsCode + "收益率计算完成")
    return res


def compare_return_rate():
    # 计算不同标注方式的收益率
    allStockCode = pd.read_csv("../QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
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
