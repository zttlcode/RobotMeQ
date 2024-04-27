import pyautogui
import time
import xlrd
import pyperclip

"""
pip install pyperclip 
pip install xlrd 
pip install pyautogui==0.9.50 
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install pillow 

3.	把每一步要操作的图标、区域截图保存至本文件夹  png格式（注意如果同屏有多个相同图标，回默认找到最左上的一个，
因此怎么截图，截多大的区域，是个学问，如输入框只截中间空白部分肯定是不行的，宗旨就是“唯一”）
4.	在cmd.xls 的sheet1 中，配置每一步的指令，如指令类型1234  对应的内容填截图文件名（别用中文），
指令5对应的内容是等待时长（单位秒） 指令6对应的内容是滚轮滚动的距离，正数表示向上滚，负数表示向下滚,数字大一点，先用200和-200试试
5.	保存文件
6.	双击waterRPA.py打开程序，按1表示excel中的指令执行一次，按2表示无限重复执行直到程序关闭
7.	如果报错不能运行用vscode运行看看报错内容（百度vscode安装与运行python程序，将报错内容xxxError后面的贴到百度上面去搜搜看）
8.	开始程序后请将程序框最小化，不然程序框挡住的区域是无法识别和操作的
9.	如果程序开始后因为你选择了无限重复而鼠标被占用停不下来，alt+F4吧~

想自己开发和优化的  可以看看pyautogui库其他用法 https://blog.csdn.net/qingfengxd1/article/details/108270159
有些同学用于操作模拟器游戏，发现鼠标移动进去无法单击双击，尝试用管理员模式运行脚本
"""
import RMQData.Asset as RMQAsset
import RMQData.HistoryData as RMQBar_HistoryData
from RMQTool import Tools as RMTTools

import os

# pyautogui库其他用法 https://blog.csdn.net/qingfengxd1/article/details/108270159


def mouseClick(clickTimes, lOrR, imgRaw, reTry):
    img = RMTTools.read_config("RMT", "tdx_img") + imgRaw
    if reTry == 1:
        while True:
            location = pyautogui.locateCenterOnScreen(img, confidence=0.9)
            if location is not None:
                pyautogui.click(location.x, location.y, clicks=clickTimes, interval=0.2, duration=0.2, button=lOrR)
                break
            print("未找到匹配图片,0.1秒后重试")
            time.sleep(0.1)
    elif reTry == -1:
        while True:
            location = pyautogui.locateCenterOnScreen(img, confidence=0.9)
            if location is not None:
                pyautogui.click(location.x, location.y, clicks=clickTimes, interval=0.2, duration=0.2, button=lOrR)
            time.sleep(0.1)
    elif reTry > 1:
        i = 1
        while i < reTry + 1:
            location = pyautogui.locateCenterOnScreen(img, confidence=0.9)
            if location is not None:
                pyautogui.click(location.x, location.y, clicks=clickTimes, interval=0.2, duration=0.2, button=lOrR)
                print("重复")
                i += 1
            time.sleep(0.1)


# 任务
def mainWork(sheet1, asset):
    i = 1
    while i < sheet1.nrows:
        # 取本行指令的操作类型
        cmdType = sheet1.row(i)[0]
        if cmdType.value == 1.0:
            # 取图片名称
            if i == 6:
                img = '4'+asset.timeLevel+'.png'
            else:
                img = sheet1.row(i)[1].value
            reTry = 1
            if sheet1.row(i)[2].ctype == 2 and sheet1.row(i)[2].value != 0:
                reTry = sheet1.row(i)[2].value
            mouseClick(1, "left", img, reTry)
            print("单击左键", img)
        # 2代表双击左键
        elif cmdType.value == 2.0:
            # 取图片名称
            img = sheet1.row(i)[1].value
            # 取重试次数
            reTry = 1
            if sheet1.row(i)[2].ctype == 2 and sheet1.row(i)[2].value != 0:
                reTry = sheet1.row(i)[2].value
            mouseClick(2, "left", img, reTry)
            print("双击左键", img)
        # 3代表右键
        elif cmdType.value == 3.0:
            # 取图片名称
            img = sheet1.row(i)[1].value
            # 取重试次数
            reTry = 1
            if sheet1.row(i)[2].ctype == 2 and sheet1.row(i)[2].value != 0:
                reTry = sheet1.row(i)[2].value
            mouseClick(1, "right", img, reTry)
            print("右键", img)
        # 4代表输入
        elif cmdType.value == 4.0:
            inputValue = str(sheet1.row(i)[1].value)

            if i == 3:
                # 输入查找代码名
                inputValue = asset.assetsCode
            elif i == 9:
                # 输入导出地址
                inputValue = RMTTools.read_config("RMQData", "tdx") + asset.assetsCode + '_' + asset.timeLevel + '.xls'
            elif i == 22:
                # 输入另存为文件名
                inputValue = asset.assetsCode + '_' + asset.timeLevel

            pyperclip.copy(inputValue)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.5)
            print("输入:", inputValue)
        # 5代表等待
        elif cmdType.value == 5.0:
            # 取图片名称
            waitTime = sheet1.row(i)[1].value
            time.sleep(waitTime)
            print("等待", waitTime, "秒")
        # 6代表滚轮
        elif cmdType.value == 6.0:
            # 取图片名称
            scroll = sheet1.row(i)[1].value
            pyautogui.scroll(int(scroll))
            print("滚轮滑动", int(scroll), "距离")
            # 6代表滚轮
        elif cmdType.value == 7.0:
            # 键盘输入34，这是通达信数据导出快捷键
            pyautogui.press('3')
            pyautogui.press('4')
            print("键盘输入34")
        i += 1


if __name__ == '__main__':
    file = RMTTools.read_config("RMT", "tdx_file") + 'cmd.xls'
    # 打开文件
    wb = xlrd.open_workbook(filename=file)
    # 通过索引获取表格sheet页
    sheet1 = wb.sheet_by_index(0)
    """
    打开通达信，调到行情页，选择前复权，只打开export文件夹，里面清空，
    想要获取数据的代码填在下面list里，运行，数据就会自动导出到QuantData文件夹里
    
    tdxList = ['510050', '159915', '510300', '510500', '512100', '588000', '159920', '159941', '512690',
               '512480', '515030', '513050', '513060', '515790', '516970', '512660', '159611', '512200',
               '512170', '512800', '512980', '512880', '515220', '159766', '159865', '518880', '159985',
               '159980', '159981', '159996', '159819', '159869', '515880', '516150', '516110', '159866',
               '513360', '513030']
    """
    # 以后需要哪些代码的数据，加到下面列表
    tdxList = ['515880', '516150', '516110', ]

    for tdx in tdxList:
        # 这里名称和资产类型无所谓
        assetList = RMQAsset.asset_generator(tdx, '', ['5', '15', '30', '60', 'd'], 'ETF')
        for asset in assetList:
            # 循环拿出每一行指令
            mainWork(sheet1, asset)
            # 数据按bar_num截断
            RMQBar_HistoryData.handle_TDX_data(asset)
            path = RMTTools.read_config("RMQData_local", "tdx")
            # 删除export导出的表格，和另存为的表格
            for filename in os.listdir(path):
                os.remove(path + filename)
            # 点一下文件夹，让它最小化
            img = sheet1.row(12)[1].value
            reTry = 1
            if sheet1.row(12)[2].ctype == 2 and sheet1.row(12)[2].value != 0:
                reTry = sheet1.row(12)[2].value
            mouseClick(1, "left", img, reTry)
            print("单击左键", img)

