import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from decimal import Decimal


# 定义fuzzy set 7个μ的分段函数
def membership(x, w1, w2, w3):  # x是x*t，w是幅度也是x轴刻度
    if x <= w1:
        y = 0
    if x > w1 and x <= w2:
        y = (x - w1) / (w2 - w1)
    if x > w2 and x <= w3:
        y = (w3 - x) / (w3 - w2)
    if x > w3:
        y = 0

    if w1 == w2:
        if x <= w2:
            y = 1
        if x > w2 and x <= w3:
            y = (w3 - x) / (w3 - w2)
        if x > w3:
            y = 0

    if w2 == w3:
        if x <= w1:
            y = 0
        if x > w1 and x <= w2:
            y = (x - w1) / (w2 - w1)
        if x > w2:
            y = 1

    return y  # 返回μ值


'''
# 以论文part2的函数代入为例，试验发现函数代入与我直接用，结果是一样的
x3 = Decimal('-0.04')
w = Decimal('0.01')

x_list = []
y_list = []

for i in range(80):
    y1 = membership(x3, 0, w, 2 * w)  # PS
    y2 = membership(x3, w, 2 * w, 3 * w)  # PM
    y3 = membership(x3, 2 * w, 3 * w, 3 * w)  # PL
    y4 = membership(x3, -2 * w, -w, 0)  # NS
    y5 = membership(x3, -3 * w, -2 * w, -w)  # NM
    y6 = membership(x3, -3 * w, -3 * w, -2 * w)  # NL
    y7 = membership(x3, -w, 0, w)  # AZ

    ed1 = 0
    y_div_1 = y1 + y2 + y3 + y7
    if y_div_1 != 0:
        ed1 = (Decimal('-0.1') * y1 - Decimal('0.2') * y2 - Decimal('0.4') * y3) / y_div_1

    ed2 = 0
    y_div_2 = y4 + y5 + y6 + y7
    if y_div_2 != 0:
        ed2 = (Decimal('0.1') * y4 + Decimal('0.2') * y5 + Decimal('0.4') * y6) / y_div_2



    ed_0 = 0
    y_div_0 = y1 + y2 + y3 + y4 + y5 + y6 + y7
    if y_div_0 != 0:
        ed_0 = (Decimal('0.1') * y1 + Decimal('0.4') * y2 - Decimal('0.2') * y3 - Decimal('0.1') * y4 - Decimal('0.4') * y5 + Decimal('0.2') * y6 + Decimal('0') * y7) / y_div_0
    x3 += Decimal('0.001')
    if x3 == Decimal('0.02'):
        print(x3)
    x_list.append(round(x3, 3))
    y_list.append(round(ed_0, 3))

plt.plot(np.array(x_list), np.array(y_list))
plt.show()
'''
'''
TODO

# 论文part3的大买家大卖家策略
def x(t, 1, n):
	p_sum = 0
	i = 0
	while i <= n-1 :
		p_sum += p[t-i]
		i += 1
	p_sum = p_sum/n
	return ln(p[t]/p_sum)  # 用ln运算，以后用math实现一下？

def ed_6(x):
	ed6 = 0
	y_div_6 = y1(x) + y2(x) + y3(x) + y7(x)
	if y_div_6 != 0:
		ed6 = (-0.1*y1(x) - 0.2*y2(x) - 0.4*y3(x))/y_div_6
	return ed6

def ed_7(x):
	ed7 = 0
	y_div_7 = y4(x) + y5(x) + y6(x) + y7(x)
	if y_div_7 != 0:
		ed7 = (0.1*y4(x) + 0.2*y5(x) + 0.4*y6(x))/y_div_7
	return ed7

def r(t+1):
	return ln(p[t+1]/p[t])

def ed_t(t)
	x = x(t, 1, n)
	return [ed_6(x), ed_7(x)].T  #  用pandas的矩阵和转置

def a_hat(t):
	if t == 0:
        return [0;0]
    else:
		return a_hat(t-1) + K(t)*(r(t+1) - ed_t(t).T * a_hat(t-1))

def K(t):
	return (P(t-1) * ed_t(t)) / (ed_t(t).T * P(t-1) * ed_t(t) + lmd)

def P(t):
	if t == 0:
		return gam
	else
		return (1 - K(t)*ed_t(t).T) * P(t-1) / lmd


# 1、初始化参数
lmd = 0.95
gam = 10

# 2、导入价格数据，  价格序列  假设是10天价格
p = [0,1,2,3,4,5,6,7,8,9,10]  # 为了方便代码可读性，p[t]就是第t天的价格，所以p[0]设置成0

# 3、计算估计参数
# 我要的结果是 二维数组，[[第一个价格的a6预测值,第一个价格的a7预测值],[第二个……,第二个……],……]
# 初始为全是0的二维数组，共n个元素
result = [[0,0],[0,0],……,[0,0]]
for t in p:
    result[t] = a_hat(t)


# 4、把a6,a7的预测值分离出来
a6_hat = []
a7_hat = []
for t in result:
    a6_hat[t] = a_hat_tmp[0]
    a7_hat[t] = a_hat_tmp[1]
'''
'''
-----------------解释一下part 3 的第三部分 参数估计算法到底啥意思---------------------

首先作者没说明白这个模拟的价格是怎么生成的，他说是基于式子（1），但是a6和a7是没有值的，所以无法生成价格

看了书才明白，作者是事先假设好了a6，a7的值（是向量），代入式子（1），生成了价格
再用生成的价格序列代入E函数下面的三个式子，得到估计参数a6，a7，与事先假设的做对比
这里有一点，使用遗忘指数的递归最小二乘法 是最小化E函数，函数具体的使用作者给了链接，我没看，以为实际使用这个算法时真的要最小化E，但实际上不用。
只需要用四个式子就行。

伪代码完成
具体实现要修改的：
1、矩阵运算与赋值
2、对数运算
3、数值精度封装
4、价格从下标1开始，最初计算时会不会越界

'''


'''
https://zhuanlan.zhihu.com/p/86297798?utm_id=0  模糊控制


INTRODUCTION 
	研究股票价格的动态变化主要有三种方法:随机游走模型,基于agent-based的模型和技术分析。
	随机行走模型的一个问题是它没有提供一个框架来研究价格形成机制的微观结构。
	agent-based模型优点是包含各种类型的交易者和交易细节，问题是参数太多分不清谁导致了某种价格特征。
	技术分析是交易者几百年的经验总结，能赚钱，但问题在于语言描述很含糊。

	这篇论文的想法是用模糊系统理论把技术分析里的规则转为“excess demand”函数——这个词的直译是超额需求，建立动态价格模型，然后通过遗忘因子最小二乘法估计参数——参数代表某种策略群体当时是否占主导。

	动态价格公式：
		市场中M个组，每个组是一种策略，每个组都有一堆交易者按照这个策略执行交易。后面PART3 （9）用到
		ai表示第i组所有人对价格的影响程度。价格动态模型的a，不是仓位，而是市场上各方的力量，谁的力量大，谁目前就是市场的主导。每个策略组都有自己的力量值a，甚至每个策略中的某个级别的交易者，也有自己的力量值，比如移动均线的短线，中线，长线。
		edi就是第i组的excess demand函数，入参xt是从过去价格和t时刻各种信息中计算出来的变量。
		M个组的值加起来，代表所有人的合力，加上当前价格，就成为了下一刻的价格。

		为什么用ln，两个价格取对数只是为了方便最后预测价格是涨还是跌。后面PART3 （9）用到

	接下来围绕这个公式做分解

	我们的目的是把技术分析里的规则转为模糊系统中的if then语句作为ed函数，这篇论文选取的规则有移动平均、支持和阻力、趋势线、带和停止、数量和相对强度、大交易者（比如机构）等，由此得到了12个启发，对应12个ed函数。

	这篇论文分为3部分，第一部分讲解模型，第二部分以移动平均为例展示模糊系统的具体细节，第三部分以香港证券市场为例讲怎么跟单机构，由此发展出的两个策略收益远超长期投资。

先讲移动平均转ed函数1
	规则：短期均线上穿（下穿）长期均线，是买入（卖出）信号。均线差距越大，涨（跌）的越强势，太强就可能短期超买（超卖），要回调，这时小卖（买）一点作为哨兵。

	定义变量：p平均t,n是均线价格，x1t反映价格变化率及趋势方向，对数ln在0~1是负数，大于1是正数，m是短期，n是长期，短期价格>长期价格，上涨趋势，比值>1，x1t为正，反之比值<1，下跌趋势，x1t为负。x1t绝对值越大，说明长短期均线差距越大，理论上x1t的值域是正负无穷。

	定义函数：我们根据x1t的取值，定义7个模糊集（fuzzy set），正大，正中等等。然后以正小，正大为例介绍了对应的分段函数，每个μ的值域都被控制在[0,1]。w是个正常量，代表价格变化百分比。

		模糊集的代码已经实现，代码逻辑就是模糊控制理论。可以搜索了解更多。
 		微信分享了个链接。

	定义买卖信号：用ed表示，需>供，为正，反之为负，也分为7个fuzzy set，买小，卖中等等，ed的取值代表仓位百分比，0.1就是用10%的仓位投资，u值也被控制在[0,1]，但是固定值。按照技术分析的买卖逻辑组合，这里Rule 15这里应该是写错了，应该是6。
	
		这个我看完了论文，发现不需要像上面的模糊集一样定义买卖仓位的μ代码，而是直接取的各个行为的最大仓位，用ci代替μ

	转为模糊系统：把结果组合成ed函数,μai是一堆[0,1]之间的值，ai代表趋势方向和程度，ci是[0,1]之间的固定值代表仓位。ed函数的返回值，就是最后合力之下交易了多少。

		根据7条规则，均线的比率影响ed1函数的决策，而一个x1t要同时进7个μ做计算，会有冲突，比如2.5w同时符合PM和PL，结果就是买0.4卖0.2
			这就是模糊控制，规则内取交集，规则之间取并集。
			等式（8）上面给出了控制理论的一些链接。Zadeh
			part2给出了具体的μ带入ed函数的方法，并给出了论文索引  (See [44], [45] for the decomposition and approximation 
foundations of fuzzy systems)

	预测：假设系数，进行3次价格预测，前100个价格由随机漫步模型生成（这个模型和动态价格公式很像，为啥都取对数呢），用价格函数预测后400个价格

支撑线阻力线转ed函数2
	规则：过去n天的最高价是阻力线，最低价是支撑线，支撑线意味着买方力量大于卖方，如果破了阻力线，开启新一波上涨，买点出现，但小突破容易失败，大突破可以买，太大的突破容易超买回调。
	resi代表阻力线，supp代表支撑线
	定义变量：当前价格和阻力线的比值是个1左右浮动的分数，对他取对数为x2t，则x2t的值域是正负无穷，为正代表突破。x3t同理，为负代表突破。
	定义函数：跟上面一样，但没写函数
	定义买卖信号：跟上面一样
	转为模糊系统：这里i=1~3是上升趋势，4~6是下跌趋势，没有中立情况（因为这不是趋势，是均值回归，所以放在下一个ed函数）。ai、ci跟上面一样

	预测：这里把两个ed函数带入 动态价格 公式。与上面的区别是有跳跃。  随机漫步模型无法模拟真实市场的跳跃，但它加了ed2就可以。

突破代表新趋势，是买卖点，但没突破就是震荡，震荡就要高抛低吸，在第一次支持线没买的交易者，会等二次回落到支撑线时买，这就是ed函数3
	规则：二次探底买中，不大是因为怕突破开启新下跌，不小是因为不想错失第二次上涨机会。
			x2t在AZ附近，就卖中。我们捋一下：AZ=0就是 当前价格：阻力线=1，对数=0。价格远离阻力线时x2t是大的负数，接近阻力线时是个小负数，此时卖中，若x2t变正数，那就该买了。均值回归和趋势是相反的操作，赔钱的核心正是分不清当前是趋势还是回归。
	定义变量：用的还是x2t，x3t
	定义函数：跟上面一样，没写函数，还是μ的
	定义买卖信号：这个就俩。
	转为模糊系统：这个ed函数与上面两个ed函数的区别是没有累加，ci写死为正负0.2，0.2是中等仓位，x2t卖中所以是-0.2，x3t买中。

	预测：把前3个edi带入 动态价格 公式。

趋势线转ed函数4
	规则：过去n天两个最高价连线或两个最低价连线，如果连线趋势向上，就是上行趋势，向下就是下行趋势。
	上行趋势看最低价连线，当价格靠近最低价连线，可以买。就是震荡上行时回踩。
	下行趋势看最高价连线，当价格靠近最高价连线，可以卖。就是震荡下行时反弹。
	为什么？因为趋势已经形成了，便很难突破。
	pup代表上升趋势，pdown代表下降趋势
	定义变量：上升趋势中，x4t为正，它接近0，说明到回踩点了。下降趋势中，x5t为负，且其绝对值靠近0，说明到反弹点了。
	定义函数：跟上面一样，我看这14个模糊集是所有策略通用的。
	定义买卖信号：x4t小正值就买中，这也没啥可说的，上升趋势它一直是正的，但很大。x5t对应。
	转为模糊系统：跟ed3类似。

	预测：前4个edi带入 动态价格公式。

趋势线没撑住，直接破了，代表趋势反转，这就是ed函数5
	规则：上升趋势，破了最低价连线中等量或大量，趋势反转，要卖。
	定义变量：还用上面的x4t,x5t
	定义函数：一样。
	定义买卖信号：上升趋势，x4t大部分是正的，变负说明价格破连线了，负很多说明破了很多，x5t同理。
	转为模糊系统：跟前面差不多。

	预测：前5个代入。

大单因为资金量大，要分成很多小部分操作，并在一段时间内慢慢增加，所以大单操作分两个：
大卖单转为ed函数6
	规则：上涨趋势卖，涨的越多卖的越多，下跌和横盘时不操作。
	定义变量：用x1t均线的
	定义买卖信号：简单，没啥可解释。
	转为模糊系统：简单。

大买单转为ed函数7	
	跟上面同理。

这里作者介绍除了大单，还有机构的操作套路，先用大买单收集筹码，然后通过新闻，吹捧，大资金快速拉升吸引趋势交易者，把股价拉上来，然后用大卖单的方法出货。这在股票市场是违法的，但几百年来全世界的市场都在用这招。

现在把快速拉升的操作转为ed函数8
	规则：无论什么情况，都大量买。
	ed6、ed7、快速拉升合并起来，就是模拟机构操作的ed函数8。

band and stop转为ed函数9
	规则：band带，这里用的布林带，注意这里提到了zero-mean，零均值，意思是一组数据的平均值等于0，通常用来描述一个随机变量或随机过程的正值和负值出现的概率应该相等。
		均线值加减2v就是布林带的上下限。
		当价格突破布林通道，意味着新趋势的开启。突破上限，则买入，突破下限，卖出。
	定义变量：x6t>0代表突破上限，x7t<0代表突破下限。
	定义买卖信号：简单
	转为模糊系统：简单
	预测：用ed1的随机漫步模型加上ed9,

止盈止损转为ed函数10
	作者提到了8种方法。这里用了一个。我觉得不太好，损失比较大。
	规则：止损是总仓位亏损20%清仓，说明策略有问题。价格在最近的高点跌下来10%，逐渐抛售。
	pbuyi是购买的价格，opbuyi是购买的股票数量，pmax是过去n天最高价。
	如果opbuyi不为空（说明有仓位），当前价格大大低于买入价格，卖大。止损。
	如果opbuyi不为空（说明有仓位），当前价格高于买入价，当前价格大大低于过去n天最高价，卖大。止盈。
		止损止盈的w系数分别设为10%，5%，
	定义变量：作者一直以来用的μ是分段函数，x*t是表示趋势和程度的对数。这里没有用x*t，而是直接用的价格和仓位。
	转为模糊系统：这里首次用了两个μ相乘。注意，若分母为0，则ed10i函数=0，ed10i代表每次卖出时的ed10，所有卖出的ed10i求个平均就是最终的ed10.注意分母不能为0，也就是必须有仓位。

部分指标转为ed函数11
	指标用来验证趋势，提示可能的趋势反转
	规则：引出了OBV指标，叫能量潮，是通过累计每日的成交量净额，制成的趋势线。计算公式是  当日OBV = 前一日OBV + 今日成交量。这个量结合价格就是顶背离和底背离。
		这里就是用的背离。价格是上升趋势，OBV是下降趋势，价格开始向下，就卖。
	定义变量：用了ed函数4的趋势线的结构（找两个最低点，最高点），求出OBV指标的上升趋势线、下降趋势线。然后求斜率，x8t就是上升趋势线斜率，x9t是下降趋势线斜率。
	定义买卖信号：上升趋势价格一直在低价连线上，所以x4t一直正，靠近0是靠近趋势线，如果变小负，说明趋势可能反转，此时obv缩量，两个高价连线斜率x9t是负数，这就是顶背离。卖中。
				 下降趋势价格一直在高价连线下，所以x5t一直是负，靠近0是靠近趋势线，如果变小正，说明趋势可能反转，此时obv放量，两个低价连线斜率x8t是正数，底背离，买中。
				 		注意，这里x8t,x9t的正负是个集合。
	转为模糊系统：买卖都是中等量，所以用正负0.2。如果分母=0，则ed11函数=0

	预测：本论文没有成交量数据，就不预测了，如果是真实市场，把ed11函数代入 （1）动态价格公式，跟之前的操作一样。

加入大盘与个股比较，转为ed函数12
	规则：个股涨不过大盘，个股稍微一跌就离开这支弱股。个股比大盘跌的少，个股稍微一涨就买它。
	rst是个股价格与大盘指数的比值。
	定义变量：x10t是当前的rst与过去n天的平均rst比值取对数
    定义买卖信号：，rst高于平均说明个股比大盘强（强就是比大盘涨的多，跌的少），取对数，就是x10t为正，如果此时是上涨趋势，不操作，如果是下跌趋势，个股一旦有涨的迹象（x5t小正——下跌趋势突破最高价连线），就买中。
										    低于平均说明个股比大盘弱，取对数，x10t为负，一旦上涨趋势有结束的迹象——x4t小负（上涨趋势破最低价连线），就卖中
    转为模糊系统：买中买中正负0.2。 若分母为0，则ed12 = 0
   
第一部分：
	总结：
		作者分析了不同策略对价格的影响，每个策略的力量在每一刻都是变化的，就像一个随机过程。所有策略的合力造成了最终价格。
		所以，按照某一个策略交易，常会有失效的时候，只是因为当时这个策略的信众力量太小。想要多个策略同时使用，实际操作时会发现不现实，因为无法预知力量系数。水太深，把握不住。
		
		当然，作者提出的策略还是有很多利用价值的，比如趋势线，大盘个股对比，可以用来筛选出强势上涨股票做趋势交易。


第二部分：
	股票短期预测靠谱。
	展示交易者行为的调整如何改变价格，a的变大导致震荡、混沌、收敛，并作出数学证明。


第三部分：
	策略
		策略很多，做哪个的信众都很鸡肋。作者最终的选择是跟机构喝汤。这个力量系数最好算，因为市场不是卖家就是卖家。


	介绍：
	作者讲两种策略，一是市场上只有大买家，它出现就买入并持有，它消失就卖；二是市场上有大买家和大卖家，买家强就买，买家弱就卖。
	第4部分会讲怎么对力量参数a进行参数估计。
	第3部分引出参数估计算法


	定义规则：分析机构的买卖时机。买家会在降价时买，涨和平时不操作。卖家会在涨价时卖，跌和平时不操作。
	这是论文1的假设7和6，所以我们假定大买家和大买家用假设6、7作为他们的交易策略。
	把两个假设转为 动态价格模型（还是PART1里那个把ed函数带入动态价格模型）
	运用从控制理论中发展出来的参数估计算法，评估动态价格模型中代表大买家、大卖家的参数。
	在大买家占优势时，大买家和大卖家都会赚钱
	我们要找到哪些股票现在处于大买家占优势的时期。

这里判断现在是大买家占优势，还是大卖家占优势，转为对a6,a7值的判断。

https://zhuanlan.zhihu.com/p/641560596?utm_id=0  使用遗忘因子最小二乘法（FFRLS）的锂离子电池二阶RC参数辨识
https://zhuanlan.zhihu.com/p/478404734?utm_id=0  参数辨识之递推最小二乘法


由于a值是随时间缓慢变化的，因此用 使用遗忘指数的递归最小二乘法  来进行参数估计
https://zhuanlan.zhihu.com/p/641560596?utm_id=0  使用遗忘因子最小二乘法（FFRLS）的锂离子电池二阶RC参数辨识
https://zhuanlan.zhihu.com/p/478404734?utm_id=0  参数辨识之递推最小二乘法

然后作者对比了估计参数和实际参数的值，发现估计参数有噪音和延后，于是定义了signal系数与noise系数比，
发现大买方/卖方对其他交易者的实力越强,对大买方/卖方的强度参数的估计越好;

四个策略回测，图示具体买卖点。
这两个策略可以用在任何资产、任何级别。

 They are easy to implement. You simply put the price data 
 
 into the parameter estimation algorithm 
(11)-(13) and make your buy/sell decisions based on the 
estimated parameters according to the simple flow-charts in 
Figs. 5 and 6; it takes just a few lines of MATLAB codes to 
implement these computations.

他们有充分的理由成功。他们不预测回报,事实上他们不预测未来的任何事情;他们只是追随大买家和大卖家的足迹:如果大买家成功,他们就会成功;如果大买家失败,他们就会失败。只要你相信供求决定价格,他们的哲学是合理的。

try other (perhaps more advanced) optimization algorithms 
to estimate the model parameters.


'''