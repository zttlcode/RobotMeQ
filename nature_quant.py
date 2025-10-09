import pandas as pd

import RMQData.Asset as RMQAsset
from RMQModel import Dataset as RMQDataset
from RMQModel import Evaluate as RMQEvaluate
from RMQModel import Label as RMQLabel
from RMQStrategy import Identify_market_types as RMQS_Identify_Market_Types
from RMQVisualized import Draw_Pyecharts as Draw_Pyecharts
import Run as Run
from sktime.datasets import write_dataframe_to_tsfile
import csv
import fnmatch
import os
import paramiko
import io
import glob
import numpy as np

from RMQStrategy import Strategy_indicator as RMQStrategyIndicator, Identify_market_types_helper as IMTHelper
from RMQTool import Message

"""
strategy_name: identify_Market_Types  label1挑出连续行情
                tea_radical_nature  
                    label1：多级别交易点合并，校验交易后日线级别涨跌幅、40个bar内趋势
                    label2：单级别校验各自涨跌幅、40个bar内趋势
                    label3：单级别校验各自MACD、DIF是否维持趋势
                    label4：单级别校验各自MACD、DIF+40个bar内趋势
                fuzzy_nature    
                    回测后，标注前，一定要process_fuzzy_trade_point_csv()预处理
                    label1 看交易对收益率
                extremum label1 看交易对收益率
                c4_oscillation_kdj_nature label1 看交易对收益率
                c4_oscillation_boll_nature label1未来价格
                c4_trend_nature label1未来价格
                c4_breakout_nature label1未来价格
                c4_reversal_nature label1未来价格                
"""


def pre_handle():
    """ """"""
    A股数据，沪深300+中证500=A股前800家上市公司，港股、美股标普500、数字币市值前9
    数据来自证券宝，每个股票5种数据：日线、60m、30m、15m、5m。日线从该股发行日到2025年1月9日，分钟级最早为2019年1月2日。
    前复权，数据已压缩备份
    
    数据回测前，先判断下载的数据有没有非法值，find_zero_close_files
    
    A股 a800_stocks  a800_stocks_wait_handle_stocks
    row['code'][3:]  row['code_name']  '5', '15', '30', '60', 'd'  stock  1  A
    
    港股 hk_1000_stock_codes  hk_1000_stock_codes_wait_handle_stocks
    row['code']  row['name']  'd'  stock  1  HK
    
    美股 sp500_stock_codes  sp500_stock_codes_wait_handle_stocks
    row['code']  row['code']  'd'  stock  1  USA
    
    数字币 crypto_code  crypto_code_wait_handle_stocks
    row['code']  row['code']  '15', '60', '240', 'd'  crypto  1  crypto
    """
    allStockCode = pd.read_csv("./QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
    # Run.parallel_backTest(allStockCode)  # 回测，并行 需要手动改里面的策略名。
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code_name'],
                                             ['30', '60', 'd'],
                                             'stock',
                                             1, 'A')

        # 回测，保存交易点,加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
        # Run.run_back_test(assetList, "tea_radical_nature")  # 0:18:27.437876 旧回测，转tick，运行时长
        # Run.run_back_test_no_tick(assetList, "fuzzy_nature", False, None)  # 0:02:29.502122 新回测，不转tick
        # RMQS_Identify_Market_Types.run_backTest_label_market_condition(assetList)  # 回测标注日线级别行情类型 该上面时间级别为d

        # 各级别交易点拼接在一起
        # concat_trade_point(assetList, "tea_radical_nature")
        # 过滤交易点
        # RMQLabel.label(assetList, "extremum", "label1")
        """
        画K线买卖点图
            method_name:
                mix: 自己在函数里自定义，用什么级别组合自己改，不需要flag
                multi_concat：多级别点位合并图，日线价格，此时flag只会是 _concat 或 _concat_label1
                single：单级别图，会用到不同过滤方式，因此flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
        """
        # Draw_Pyecharts.show(assetList, "single", "tea_radical_nature", None)
        # Draw_Pyecharts.show(assetList, "single", "tea_radical_nature", "_label2")

        """
        计算收益率
            is_concat: True 计算合并交易点的收益率  此时flag只会是 _concat 或 _concat_label1
                       False 计算各个级别，此时flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
            pred：True会读取模型预测结果，并以此计算收益
            pred_tpp：True会读取模型二次过滤的结果
            handled_uneven：True会用均样本之后的数据算收益。把均样本之后的数据存到本地，方便计算原始收益和模型预测收益
        """
        # res1 = RMQEvaluate.return_rate(assetList, False, None, "tea_radical_nature",
        #                                False, False, False)


def prepare_train_dataset():
    """"""
    """
    标注完成，准备训练数据
    由于两个数据集都要做，因此写俩方法串行，别删
        按照原始标注方法，_TRAIN 最多24.6万  _TEST 最多14.6万
        limit_length==0 代表不截断，全数据
        
        flag: _TRAIN 训练集截前500个股票，  _TEST 测试集截后300个  截之前按固定随机数乱序了
        time_point_step: 截取的时间步，最长500，最少得是8以上，因为很多时序模型需要得序列长度最少是8
        limit_length：限制长度是为了方便debug时调试，数据太多加载太慢
        handle_uneven_samples: macd策略样本不均，其他策略不一定有这个问题，所以这里控制要不要处理
        strategy_name: 为了读回测点文件，     
                        identify_Market_Types  不需要处理样本不均
                        fuzzy_nature
                        tea_radical_nature
                        c4_trend_nature
                        c4_oscillation_boll_nature
                        c4_oscillation_kdj_nature
                        c4_breakout_nature
                        c4_reversal_nature
                        extremum  不需要处理样本不均
        feature_plan_name: 不同特征组织方案
                feature_tea_concat 日线、小时线、指数日线
                feature_tea_multi_level macd5分钟、15分钟、30分钟
                feature_fuzzy_multi_level
                feature_all
                feature_tea_macd
                feature_c4_trend
                feature_c4_oscillation_boll
                feature_c4_oscillation_kdj
                feature_c4_breakout
                feature_c4_reversal  
                feature_extremum
        p2t_name: point_to_ts_single
                  point_to_ts_up_time_level
                  point_to_ts_concat  拼接所有点，60、d、index_d造数
                  point_to_ts_tea_multi_level  5标注，5、15、30造数
                  point_to_ts_fuzzy_multi_level  30标注，30、60、d造数
        label_name: 合并标注交易点的  此时flag只会是 _concat_label1
                    各级别标注交易点  "_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
                    fuzzy的各级别flag也有 _label1
        name: 标的_级别 ts文件命名，跟limit_length对应，这文件有多少条数据
                跑单级别时，在Dataset里只填对应级别        
    """
    # RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
    #                            50000, True,
    #                            "extremum", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")
    # RMQDataset.prepare_dataset("_TEST", "A_15", 160,
    #                            10000, True,
    #                            "extremum", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")

    # RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
    #                            50000, True,
    #                            "c4_oscillation_boll_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")
    # RMQDataset.prepare_dataset("_TEST", "A_15", 160,
    #                            10000, True,
    #                            "c4_oscillation_boll_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")

    RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
                               50000, True,
                               "tea_radical_nature", "feature_all",
                               "point_to_ts_single", "_label2", 2, "buy")
    RMQDataset.prepare_dataset("_TEST", "A_15", 160,
                               10000, True,
                               "tea_radical_nature", "feature_all",
                               "point_to_ts_single", "_label2", 2, "buy")

    # RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
    #                            50000, True,
    #                            "fuzzy_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")
    # RMQDataset.prepare_dataset("_TEST", "A_15", 160,
    #                            10000, True,
    #                            "fuzzy_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")
    #
    # RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
    #                            50000, True,
    #                            "c4_oscillation_kdj_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")
    # RMQDataset.prepare_dataset("_TEST", "A_15", 160,
    #                            10000, True,
    #                            "c4_oscillation_kdj_nature", "feature_all",
    #                            "point_to_ts_single", "_label1", 2, "buy")


def prepare_pred_dataset():
    """""" """
    组装预测数据  
        预测交易点:
            point_to_ts_single  用本级别交易点，找对应时间回测数据，策略和特征可随意组合
            pred_market_type False
            up_time_level 任意值都行
        预测行情
            point_to_ts_up_time_level  用本级别交易点，找up_time_level对应级别的回测数据，
            特征只用feature_all
    
    20251003 处理c4_oscillation_boll_nature 和 c4_oscillation_kdj_nature时报文件不存在，原因是文件夹名字太长了，在Dataset.py
    里把策略名改短即可
    """
    # RMQDataset.prepare_dataset_single("_TEST", "A_15", 160,
    #                                   20000, True,
    #                                   "fuzzy_nature", "feature_all",
    #                                   "point_to_ts_single", "_label1", None,
    #                                   False, 'd')

    # 预测收益
    allStockCode = pd.read_csv("D:/github/RobotMeQ/QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
    df_dataset = allStockCode.iloc[500:]
    n = 1
    count = 0
    count_win = 0
    for index, row in df_dataset.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             '',
                                             ['15'],
                                             'stock',
                                             1, 'A')
        # 标注收益，这是最完美结果
        res1 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_boll_nature",
                                False, False, True)
        # 模型预测
        res2 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_boll_nature",
                                True, False, True)
        # 多模型过滤，比模型预测收益高就好。
        # RMQEvaluate.return_rate(assetList, False, "_label2", "tea_radical_nature",
        #                         True, True, True)
        if res1 is None or res2 is None:
            continue
        if res1 < res2:
            count_win += 1
        count += 1
        # print(f"{asset.assetsCode}{flag} 最终结果 持股数: {shares}, 市值: {holding_value:.2f}, "
        #       f"总投资额: {latest_total_cost:.2f}, 持股收益率: {final_return_rate:.2%}, "
        #       f"总收益: {(holding_value-latest_total_cost):.2f}")
        print(f"{assetList[0].assetsCode} 原始策略收益率: {res1:.2%}, "
              f"标注策略收益率: {res2:.2%}", f"收益率提升: {res2-res1:.2%}")
        n += 1
        if n > 20:
            break
    print(f"{count_win/count:.2%}")
    print(count_win, count)


def run_live():
    # 服务器信息
    server_ip = "192.168.0.102"
    username = "root"
    password = "zhao1993"
    docker_container_id = "d63f10ba76df"
    live_dir = "/home/RobotMeQ/QuantData/live_to_ts/"  # Docker 内的 CSV 目录
    local_backup_dir = "D:/github/RobotMeQ/QuantData/live_to_ts/"  # 本地备份目录

    # 连接服务器
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server_ip, username=username, password=password)

    # 获取 `live` 目录下的所有 CSV 文件
    command_list_csv = f"docker exec {docker_container_id} ls {live_dir} | grep '.csv'"
    stdin, stdout, stderr = client.exec_command(command_list_csv)
    csv_files = stdout.read().decode().splitlines()

    if not csv_files:
        print("没有找到任何 CSV 文件")
    else:
        for csv_file in csv_files:
            csv_path = f"{live_dir}{csv_file}"
            print(f"正在处理文件: {csv_path}")

            # **Step 1: 备份 CSV 文件**
            backup_path = f"/tmp/{csv_file}"  # 先拷贝到服务器宿主机
            command_backup = f"docker cp {docker_container_id}:{csv_path} {backup_path}"
            client.exec_command(command_backup)

            # **Step 2: 下载文件到本地**
            sftp = client.open_sftp()
            local_file_path = os.path.join(local_backup_dir, csv_file)
            sftp.get(backup_path, local_file_path)  # 下载文件到本地
            sftp.close()
            print(f"已备份到本地: {local_file_path}")

            # 读取 CSV 文件的第一行（存储的是 Python 列表格式的字符串）
            command_read_csv = f"docker exec {docker_container_id} head -n 1 {csv_path}"
            stdin, stdout, stderr = client.exec_command(command_read_csv)
            first_line = stdout.read().decode().strip()

            if not first_line:
                print(f"文件 {csv_file} 为空，跳过处理")
                continue

            data = first_line.split(",")
            # **调用 asset_generator**
            assetList = RMQAsset.asset_generator(data[3],
                                                 data[4],
                                                 [data[5]],
                                                 'ETF' if data[3].startswith('1') or data[3].startswith(
                                                     '5') else 'stock',
                                                 1, 'A')
            # **获取 live_bar 文件路径**
            # live_bar = "/home/RobotMeQ/QuantData/live/live_bar_A_"+data[3]+"_"+data[5]+".csv"
            # live_bar = "/home/RobotMeQ/QuantData/live/live_bar_159611_15.csv"

            # # **读取 live_bar 文件内容**
            command_read_live_bar = f"docker exec {docker_container_id} cat {csv_path}"
            stdin, stdout, stderr = client.exec_command(command_read_live_bar)
            live_bar_content = stdout.read().decode()

            # **用 Pandas 解析 live_bar 数据**
            try:
                data_0 = pd.read_csv(io.StringIO(live_bar_content), index_col="time", parse_dates=True)
            except Exception as e:
                print(f"无法解析 {csv_path}: {e}")

            # 准备ts数据
            data_0 = IMTHelper.calculate_indicators(data_0)
            # 处理Nan
            data_0.fillna(method='bfill', inplace=True)  # 用后一个非NaN值填充（后向填充）
            data_0.fillna(method='ffill', inplace=True)  # 用前一个非NaN值填充（前向填充）

            data_0_tmp = data_0.iloc[-160:].reset_index(drop=True)
            ema10 = data_0_tmp["ema10"]
            ema20 = data_0_tmp["ema20"]
            ema60 = data_0_tmp["ema60"]
            macd = data_0_tmp["macd"]
            signal = data_0_tmp["signal"]
            adx = data_0_tmp["adx"]
            plus_di = data_0_tmp["plus_di"]
            minus_di = data_0_tmp["minus_di"]
            atr = data_0_tmp["atr"]
            boll_mid = data_0_tmp["boll_mid"]
            boll_upper = data_0_tmp["boll_upper"]
            boll_lower = data_0_tmp["boll_lower"]
            rsi = data_0_tmp["rsi"]
            obv = data_0_tmp["obv"]
            volume_ma5 = data_0_tmp["volume_ma5"]
            close = data_0_tmp["close"]
            volume = data_0_tmp["volume"]

            temp_data_dict = {'ema10': [], 'ema20': [], 'ema60': [], 'macd': [], 'signal': [],
                              'adx': [], 'plus_di': [], 'minus_di': [], 'atr': [], 'boll_mid': [], 'boll_upper': [],
                              'boll_lower': [], 'rsi': [], 'obv': [], 'volume_ma5': [], 'close': [], 'volume': []
                              }

            temp_data_dict['ema10'].append(ema10)
            temp_data_dict['ema20'].append(ema20)
            temp_data_dict['ema60'].append(ema60)
            temp_data_dict['macd'].append(macd)
            temp_data_dict['signal'].append(signal)
            temp_data_dict['adx'].append(adx)
            temp_data_dict['plus_di'].append(plus_di)
            temp_data_dict['minus_di'].append(minus_di)
            temp_data_dict['atr'].append(atr)
            temp_data_dict['boll_mid'].append(boll_mid)
            temp_data_dict['boll_upper'].append(boll_upper)
            temp_data_dict['boll_lower'].append(boll_lower)
            temp_data_dict['rsi'].append(rsi)
            temp_data_dict['obv'].append(obv)
            temp_data_dict['volume_ma5'].append(volume_ma5)
            temp_data_dict['volume'].append(volume)
            temp_data_dict['close'].append(close)

            temp_label_list = []
            if data[2] == "buy":
                temp_label_list.append("1")
            else:
                temp_label_list.append("3")

            # 循环结束后，字典转为DataFrame
            result_df = pd.DataFrame(temp_data_dict)
            # 将列表转换成 Series
            result_series = pd.Series(temp_label_list)

            strategy_name = "tea_radical"  # tea_radical fuzzy
            feature_plan_name = "feature_all"
            problem_name_str = ("pred_live_" + assetList[0].assetsMarket + "_"
                                + assetList[0].assetsCode + "_" + assetList[0].barEntity.timeLevel
                                + "_" + str(strategy_name) + "_" + str(feature_plan_name) + "_160_step")
            class_value_list_str = ["1", "2", "3", "4"]
            # 写入 ts 文件
            write_dataframe_to_tsfile(
                data=result_df,
                path="D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts/prediction_live",  # 保存文件的路径
                problem_name=problem_name_str,  # 问题名称
                class_label=class_value_list_str,  # 是否有 class_label
                class_value_list=result_series,  # 是否有 class_label
                equal_length=True,
                fold="_TEST"
            )
            # 写入 ts 文件
            write_dataframe_to_tsfile(
                data=result_df,
                path="D:/github/RobotMeQ_Dataset/QuantData/trade_point_backTest_ts/prediction_live",  # 保存文件的路径
                problem_name=problem_name_str,  # 问题名称
                class_label=class_value_list_str,  # 是否有 class_label
                class_value_list=result_series,  # 是否有 class_label
                equal_length=True,
                fold="_TRAIN"
            )
            break
    # **Step 4: 删除 Docker 容器内的所有 CSV 文件**
    command_delete_csv = f"docker exec {docker_container_id} rm -f {live_dir}*.csv"
    client.exec_command(command_delete_csv)
    print("所有 Docker 容器中的 CSV 文件已删除")

    # **Step 5: 删除服务器 `/tmp` 目录下的备份 CSV 文件**
    command_delete_tmp_csv = f"rm -f /tmp/*.csv"
    client.exec_command(command_delete_tmp_csv)
    print("所有服务器 `/tmp` 目录中的备份 CSV 文件已清空")
    # **关闭 SSH 连接**
    client.close()


def run_live_get_pred():
    local_backup_dir = "D:/github/RobotMeQ/QuantData/live_to_ts/"  # 本地备份目录

    # 确保目录存在
    if not os.path.exists(local_backup_dir):
        print(f"本地备份目录 {local_backup_dir} 不存在")
    else:
        # 遍历目录下的所有 CSV 文件
        for filename in os.listdir(local_backup_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(local_backup_dir, filename)
                print(f"正在读取: {file_path}")

                # 读取 CSV 文件的第一行
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        data = next(reader)  # 读取第一行

                        df_prd_true_filePath = "D:/github/Time-Series-Library-Quant/results/" + data[
                            3] + "_prd_result.csv"
                        if not os.path.exists(df_prd_true_filePath):
                            print(data[3] + "预测结果文件不存在")
                            break
                        df_prd_true = pd.read_csv(df_prd_true_filePath)
                        if df_prd_true['trues'] == df_prd_true['predictions']:
                            # 发消息
                            post_msg = (data[4]
                                        + "-"
                                        + data[3]
                                        + "-"
                                        + data[5]
                                        + "：" + data[2] + "："
                                        + str(data[1])
                                        + " 时间："
                                        + data[0])
                            mail_msg = Message.build_msg_text_no_entity("macd+kdj", post_msg)
                            mail_list_qq = "mail_list_qq_d"
                            res = Message.QQmail(mail_msg, mail_list_qq)
                            if res:
                                print('发送成功')
                            else:
                                print('发送失败')
                    except StopIteration:
                        print(f"文件 {filename} 为空，无法读取第一行")
                    except IndexError:
                        print(f"文件 {filename} 第一行列数不足，无法读取 data[3]")
    print("所有文件读取完成")
    # 获取所有文件
    files1 = glob.glob(os.path.join(local_backup_dir, "*"))
    # 遍历删除
    for file in files1:
        os.remove(file)  # 删除文件

    files2 = glob.glob(os.path.join("D:/github/Time-Series-Library-Quant/results/", "*"))
    # 遍历删除
    for file in files2:
        os.remove(file)  # 删除文件


def run_live_run():
    from time import sleep
    from datetime import datetime, time
    from RMQTool import Tools as RMTTools

    while True:
        # 只能交易日的0~9:30之间，或交易日15~0之间，手动启
        workday_list = RMTTools.read_config("RMT", "workday_list") + "workday_list.csv"
        result = RMTTools.isWorkDay(workday_list, datetime.now().strftime("%Y-%m-%d"))  # 判断今天是不是交易日  元旦更新
        if result:  # 是交易日
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "早上开启进程")

            while (time(9, 30) < datetime.now().time() < time(11, 34)
                   or time(13) < datetime.now().time() < time(15, 4)):
                run_live()
                run_live_get_pred()
                sleep(360)  # 每过5分钟+60秒，执行一次

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "中午进程停止，等下午")
            sleep(1800)  # 11:30休盘了，等半小时到12:30，开下午盘
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午开启进程")

            while (time(9, 30) < datetime.now().time() < time(11, 34)
                   or time(13) < datetime.now().time() < time(15, 4)):
                run_live()
                run_live_get_pred()
                sleep(360)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "下午进程停止，等明天")
            sleep(61200)  # 15点收盘，等17个小时，到第二天8点，重新判断是不是交易日
        else:  # 不是交易日
            sleep(86400)  # 直接等24小时，再重新判断


def pre_handle_compare_label_profit(strategy):
    allStockCode = pd.read_csv("./QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
    count = 0
    count_win = 0
    # 存储差值的列表
    differences = []
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'][3:],
                                             row['code'],
                                             ['d'],
                                             'stock',
                                             1, 'A')
        res1 = RMQEvaluate.return_rate(assetList, False, None, strategy,
                                       False, False, False)
        if strategy == 'tea_radical_nature':
            flag = "_label2"
        else:
            flag = "_label1"
        res2 = RMQEvaluate.return_rate(assetList, False, flag, strategy,
                                       False, False, False)
        if res1 is None or res2 is None:
            continue
        if res1 < res2:
            count_win += 1
        count += 1
        """
        res1 < res2时，如果两个都是正数，diff是收益率提升了百分之多少，这没问题。
        如果res1是负数，res2是正数，差是正数，但被除数是负数
        如果res1是负数，res2是负数，差是正数，但被除数是负数
        """
        if res2 == res1:
            diff = 0
        elif 0 <= res1 < res2:
            if res1 == 0:
                diff = res2
            else:
                diff = (res2 - res1) / res1  # 计算差值
        elif res1 < 0 <= res2:
            diff = (res2 - res1) / abs(res1)
        elif res1 < res2 < 0:
            diff = (res2 - res1) / abs(res1)

        if 0 <= res2 < res1:
            diff = (res2 - res1) / res1
        elif res2 < 0 <= res1:
            if res1 == 0:
                diff = res2
            else:
                diff = (res2 - res1) / res1
        elif res2 < res1 < 0:
            diff = (res2 - res1) / abs(res1)

        differences.append(diff)
        break
    print("-----------" + strategy + "开始---------")
    print(f"{count_win/count:.2%}")
    print(count_win, count)

    def exclude_extremes(differences):
        # 计算要排除的极值数量（总数的1%，即0.5%最小和0.5%最大）
        n = len(differences)
        k = int(n * 0.05)  # 5% 的数量

        if k == 0:
            print("数据太少")
            return differences.copy()  # 如果数据量太少，直接返回副本

        # 排序列表以便找到极值
        sorted_diff = sorted(differences)

        # 最小的k个和最大的k个值
        min_extremes = set(sorted_diff[:k])
        max_extremes = set(sorted_diff[-k:])

        # 排除这些极值后的列表
        filtered = [x for x in differences if x not in min_extremes and x not in max_extremes]

        return filtered
    # 计算平均差值
    average_diff_raw = np.mean(differences)
    # 计算差值的中位数
    median_diff_raw = np.median(differences)
    print(f"原始平均差值: {average_diff_raw:.2%}，正数表示标注有提升")
    print(f"原始差值中位数: {median_diff_raw:.2%}，正数表示标注有提升")

    filtered_differences = exclude_extremes(differences)
    # 计算平均差值
    average_diff = np.mean(filtered_differences)
    # 计算差值的中位数
    median_diff = np.median(filtered_differences)
    print(f"95%平均差值: {average_diff:.2%}，正数表示标注有提升")
    print(f"95%差值中位数: {median_diff:.2%}，正数表示标注有提升")

    print(len(differences), len(filtered_differences))
    # 确保 differences 是一维NumPy数组
    filtered_differences = np.array(filtered_differences)
    positive_values = filtered_differences[filtered_differences > 0]
    print(len(positive_values), len(filtered_differences))
    if len(positive_values) > 0:
        mean_neg = np.mean(positive_values)
        median_neg = np.median(positive_values)
        print(f"提升的均值: {mean_neg:.2%}")
        print(f"提升的中位数: {median_neg:.2%}")
    else:
        print("数组中没有正数！")
    print("-----------"+strategy+"结束---------")


def pre_handle_compare_strategy():
    allStockCode = pd.read_csv("./QuantData/asset_code/sp500_stock_codes.csv", dtype={'code': str})
    # 初始化各策略总得分数组（7个策略）
    total_profit = np.zeros(8)
    count = 0
    for index, row in allStockCode.iterrows():
        assetList = RMQAsset.asset_generator(row['code'],
                                             row['code'],
                                             ['d'],
                                             'stock',
                                             1, 'USA')
        res1 = RMQEvaluate.return_rate(assetList, False, "_label2", "tea_radical_nature",
                                       False, False, False)
        res2 = RMQEvaluate.return_rate(assetList, False, "_label1", "fuzzy_nature",
                                       False, False, False)
        res3 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_kdj_nature",
                                       False, False, False)
        res4 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_oscillation_boll_nature",
                                       False, False, False)
        res5 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_trend_nature",
                                       False, False, False)
        res6 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_breakout_nature",
                                       False, False, False)
        res7 = RMQEvaluate.return_rate(assetList, False, "_label1", "c4_reversal_nature",
                                       False, False, False)
        res8 = RMQEvaluate.return_rate(assetList, False, "_label1", "extremum",
                                       False, False, False)
        # 将结果存入数组
        profits = np.array([res1, res2, res3, res4, res5, res6, res7, res8], dtype=float)

        # 检查数据有效性
        if not np.all(np.isfinite(profits)):
            print(f"跳过无效数据: {profits}", row['code'])
            continue

        # 直接累加收益金额
        total_profit += profits
        count += 1
    total_profit = total_profit/count
    # 按总收益排序
    strategies = ["tea_radical_nature", "fuzzy_nature", "c4_oscillation_kdj_nature",
                  "c4_oscillation_boll_nature", "c4_trend_nature", "c4_breakout_nature",
                  "c4_reversal_nature", "extremum"]
    sorted_indices = np.argsort(-total_profit)

    print("策略总收益排名：")
    for rank, idx in enumerate(sorted_indices, 1):
        # print(f"第{rank}名: {strategies[idx]}，总收益：{total_profit[idx]:.2f}元")
        # 2025 05 06 收益金额改收益率
        print(f"第{rank}名: {strategies[idx]}，总收益率：{total_profit[idx]:.2f}%")


if __name__ == '__main__':
    """
    报错：
    前复权会导致很久以前的股价出现负数，坑啊
    行情分类做ts时忘记改为3类了
    ValueError: Expected input batch_size (1) to match target batch_size (0). 这个报错是数据量运气不好，随便改一下数据量就好了
    C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\Loss.cu:250: block: [0,0,0], thread: [8,0,0] Assertion `t >= 0 && t < n_classes` failed.
    这个报错是分类有问题，要么模型最后的分类不是四类，要么ts文件里label有Nan导致分类不是4类
    fuzzy,kdj,极值，这三个回测后都是交易对，要先预处理，再标注，不然标注会出现Nan，训练时会出现Nan类
    run里面多个策略同时跑，导致订单出现错误，数据混乱
    point_to_ts_single 是找当前级别数据，如果某一行数据有问题直接跳过
        point_to_ts_up_time_level 对分钟级来说，是找上级，因此本级别找的行数不同，原来有问题的行可能被跳过。因此导致预测行数与行情预测时的行数对不上
        解决办法只能在计算完指标后，填充Nan
    """
    # pre_handle()  # 数据预处理
    # pre_handle_compare_strategy()  # 预处理中统计各原始策略的收益排名
    # strategies = ["c4_breakout_nature", "c4_oscillation_boll_nature", "c4_trend_nature", "tea_radical_nature",
    #               "c4_reversal_nature", "fuzzy_nature", "c4_oscillation_kdj_nature"]
    # for strategy in strategies:
    #     pre_handle_compare_label_profit(strategy)  # 预处理中统计各策略标注后收益是否提升
    # pre_handle_compare_label_profit("c4_oscillation_kdj_nature")  # 预处理中统计各策略标注后收益是否提升
    prepare_train_dataset()  # 所有股票组成训练集
    # prepare_pred_dataset()  # 单独推理一个股票
    # run_live()
    pass
