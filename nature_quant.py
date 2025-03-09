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


from RMQStrategy import Strategy_indicator as RMQStrategyIndicator, Identify_market_types_helper as IMTHelper
from RMQTool import Message


def pre_handle():
    """ """"""
    A股数据，沪深300+中证500=A股前800家上市公司，港股、美股标普500、数字币市值前9
    数据来自证券宝，每个股票5种数据：日线、60m、30m、15m、5m。日线从该股发行日到2025年1月9日，分钟级最早为2019年1月2日。前复权，数据已压缩备份
    
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
        # assetList = RMQAsset.asset_generator(row['code'][3:],
        #                                      row['code_name'],
        #                                      ['5', '15', '30', '60', 'd'],
        #                                      'stock',
        #                                      1, 'A')
        assetList = RMQAsset.asset_generator('601658',
                                             '601658',
                                             ['5', '15', '30', '60', 'd'],
                                             'stock',
                                             1, 'A')
        # 回测，保存交易点,加tick会细化价格导致操作提前，但实盘是bar结束了算指标，所以不影响
        # Run.run_back_test(assetList, "tea_radical_nature")  # 0:18:27.437876 旧回测，转tick，运行时长
        # Run.run_back_test_no_tick(assetList, "fuzzy_nature")  # 0:02:29.502122 新回测，不转tick
        # RMQS_Identify_Market_Types.run_backTest_label_market_condition(assetList)  # 回测标注日线级别行情类型 该上面时间级别为d

        # 各级别交易点拼接在一起
        # concat_trade_point(assetList, "tea_radical_nature")
        """
        过滤交易点
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
        # RMQLabel.label(assetList, "fuzzy_nature", "label1")
        """
        画K线买卖点图
            method_name:
                mix: 自己在函数里自定义，用什么级别组合自己改，不需要flag
                multi_concat：多级别点位合并图，日线价格，此时flag只会是 _concat 或 _concat_label1
                single：单级别图，会用到不同过滤方式，因此flag有2种，
                        原始交易点："_" + asset.barEntity.timeLevel  此时flag是 None
                        各级别标注交易点："_" + asset.barEntity.timeLevel + "_label3"  此时flag是 _label2 _label3 _label4
        """
        #  Draw_Pyecharts.show(assetList, "single", "c4_trend_nature", "_label1")
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
        RMQEvaluate.return_rate(assetList, False, None, "fuzzy_nature",
                                False, False, False)
        RMQEvaluate.return_rate(assetList, False, "_label1", "fuzzy_nature",
                                False, False, False)
        break


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
    RMQDataset.prepare_dataset("_TRAIN", "A_15", 160,
                               20000, True,
                               "tea_radical_nature", "feature_all",
                               "point_to_ts_single", "_label3")
    RMQDataset.prepare_dataset("_TEST", "A_15", 160,
                               10000, True,
                               "tea_radical_nature", "feature_all",
                               "point_to_ts_single", "_label3")


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
    """
    RMQDataset.prepare_dataset_single("_TEST", "A_15", 160,
                                      20000, True,
                                      "tea_radical_nature", "feature_all",
                                      "point_to_ts_up_time_level", "_label3", 20,
                                      True, 'd')

    # 预测收益
    # allStockCode = pd.read_csv("D:/github/RobotMeQ/QuantData/asset_code/a800_stocks.csv", dtype={'code': str})
    # df_dataset = allStockCode.iloc[500:]
    # n = 1
    # for index, row in df_dataset.iterrows():
    #     assetList = RMQAsset.asset_generator(row['code'][3:],
    #                                          '',
    #                                          ['15'],
    #                                          'stock',
    #                                          1, 'A')
    #     # 标注收益，这是最完美结果
    #     RMQEvaluate.return_rate(assetList, False, "_label3", "tea_radical_nature",
    #                             False, False, True)
    #     # 模型预测
    #     RMQEvaluate.return_rate(assetList, False, "_label3", "tea_radical_nature",
    #                             True, False, True)
    #     # 多模型过滤，比模型预测收益高就好。
    #     RMQEvaluate.return_rate(assetList, False, "_label3", "tea_radical_nature",
    #                             True, True, True)
    #     n += 1
    #     if n > 20:
    #         break


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
            live_bar = "/home/RobotMeQ/QuantData/live/live_bar_159611_15.csv"

            # # **读取 live_bar 文件内容**
            command_read_live_bar = f"docker exec {docker_container_id} cat {live_bar}"
            stdin, stdout, stderr = client.exec_command(command_read_live_bar)
            live_bar_content = stdout.read().decode()

            # **用 Pandas 解析 live_bar 数据**
            try:
                data_0 = pd.read_csv(io.StringIO(live_bar_content), index_col="time", parse_dates=True)
            except Exception as e:
                print(f"无法解析 {live_bar}: {e}")

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

                        df_prd_true_filePath = "D:/github/Time-Series-Library-Quant/results/" + data[3] + "_prd_result.csv"
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
                            mail_msg = Message.build_msg_text_no_entity("macd+kdj",post_msg)
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


if __name__ == '__main__':
    pre_handle()  # 数据预处理
    # prepare_train_dataset()  # 所有股票组成训练集
    # prepare_pred_dataset()  # 单独推理一个股票
    # run_live()
    pass
