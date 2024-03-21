import os
from configparser import ConfigParser
import requests
import json
import pandas as pd

# 读取各种配置文件的方法 https://blog.csdn.net/qq_62789540/article/details/127188981


def read_config(section, item):
    cp = ConfigParser()
    # 这里必须写绝对路径，如果写相对路径，会去找调用这个函数的python文件的相对路径，而不是当前文件的相对路径
    # 这里必须写绝对路径，如果写相对路径，会去找调用这个函数的python文件的相对路径，而不是当前文件的相对路径
    # 发布服务器时，只改这里，然后运行对应的run就可以，其他任何代码都不用动
    path = "D:\\workspace\\github\\RobotMeQ\\Configs\\config.ini"
    # path = "/home/RobotMeQ/Configs/config_prd.ini"
    cp.read(path, encoding='utf-8')
    return cp.get(section, item)


def write_config(section, item, value):
    cp = ConfigParser()
    # 这里必须写绝对路径，如果写相对路径，会去找调用这个函数的python文件的相对路径，而不是当前文件的相对路径
    # 发布服务器时，只改这里，然后运行对应的run就可以，其他任何代码都不用动
    path = "D:\\workspace\\github\\RobotMeQ\\Configs\\config.ini"
    # path = "/home/RobotMeQ/Configs/config_prd.ini"
    cp.read(path, encoding='utf-8')
    cp.set(section, item, value)
    with open(path, "w") as configfile:
        cp.write(configfile)


def getWorkDay():
    """
    在深交所官网，通过F12，找到交易日历的链接，通过循环传入1~12月，返回当月所有日期，
    其中jybz==‘1’说明是交易日，加入list
    12个月全部查完后，把list转为Dataframe，写入csv
    """
    workday_list = []
    month = 1
    while month < 13:
        res = requests.get("http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month=2024-"+str(month))
        dic = json.loads(res.text)
        for dayDic in dic["data"]:
            if dayDic['jybz'] == '1':
                workday_list.append(dayDic['jyrq'])
        month = month + 1
    df = pd.DataFrame(workday_list, columns=['workday'])
    df.to_csv("D:\\workspace\\github\\RobotMeQ\\QuantData\\workday_list.csv", index=False)


def isWorkDay(filepath, today):
    # 读取工作日列表，遍历看今天是不是交易日
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        if today == row['workday']:
            return True


if __name__ == '__main__':
    getWorkDay() # 获取今年的交易日。需要每年元旦运行
    # sss = read_config("RMT", "mail_list_qq")
    # print(sss)


