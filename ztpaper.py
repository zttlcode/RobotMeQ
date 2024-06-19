import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch import nn
from torchvision import transforms


def linear_regression():
    # 读取原始数据
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')

    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    # torch.tensor( dataframe类型.to_numpy(dtype=float) )  矩阵的列名会去掉，并转为张量

    # 本例中，价格几十，成交量几千万
    # 原始数据不做任何预处理，每次epoch打印损失时，报nan，可能是特征之间数据之间差距过大，通过特征缩放矫正
    # 且差距大会导致权重不均，收敛慢
    # 特征缩放有两种方式：
    # 1、标准化，将特征缩放到均值为0，标准差为1的分布，没有范围
    # mean = features.mean(dim=0)  # 计算每个特征的均值
    # std = features.std(dim=0)  # 计算每个特征的标准差
    # eps = 1e-8  # 为了避免除以零的情况，我们可以设置一个小的正数作为分母
    # features = (features - mean) / (std + eps)  # 标准化数据：每个特征值减去其均值并除以其标准差

    # 2、最大最小值，范围固定在[0,1]
    min_vals = torch.min(features, dim=0)[0]  # 计算每列特征的最小值和最大值
    max_vals = torch.max(features, dim=0)[0]
    features = (features - min_vals) / (max_vals - min_vals)  # 对每个特征进行最小-最大缩放

    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)  # 标签不做处理

    # 构建随机小批量迭代器
    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    data_iter = load_array((features, labels), batch_size)

    # 内部协变量偏移问题指的是在训练过程中，由于每一层神经网络的参数不断更新，导致每一层输入的分布也会随之发生变化。
    # 这种变化进而会影响下一层的训练，使其变得更加困难，可能需要花费更长的时间来训练，因此要通过批量归一化，
    # 这样，无论网络的前一层输出什么样的分布，BatchNorm都能保证当前层的输入具有合适的尺度
    # 比如输入是2个特征，先用批量归一化分别把两列特征改成均值为0，方差为1，然后2个特征再进入线性层
    # 批量归一化不改变范围，例如批量归一化后，价格变成20左右，成交量变成500万左右
    # 但我只有一层，没有必要用这个
    # net = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
    net = nn.Sequential(nn.Linear(2, 1))
    # Linear在第二位，所以要操作net[1]
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 5
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 通过调用net(X)生成预测并计算损失l（前向传播）
            trainer.zero_grad()
            l.backward()  # 通过进行反向传播来计算梯度  算出每个系数的偏导
            trainer.step()  # 通过调用优化器来更新模型参数  系数 = 系数 - 梯度*学习率  batchsize放在了损失函数里
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1},loss{l:f},p_price{net(features)[-1, 0]:f},true_price{labels[-1, 0]:f}')


# 批量归一化  上面解释了，多层要控制输出才需要用
def linear_regression_batchnorm():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    # 定义均值和标准差
    mean = [0.5, 0.5]  # 特征的均值，假设有两个特征
    std = [0.5, 0.5]  # 特征的标准差，假设有两个特征

    # 定义归一化变换
    normalize = transforms.Normalize(mean=mean, std=std)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        # 创建 Subset，只包含特征
        subset_indices = list(range(len(features)))
        subset = data.Subset(dataset, subset_indices)  # 这就能单独取到features？
        # 对数据集进行归一化处理，仅应用于特征
        subset = subset.transform(transform=normalize)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    # 在这个示例中，我们首先创建了TensorDataset对象，并将特征和标签传递给它。
    # 然后，我们使用Subset来创建一个只包含特征的子集。
    # 最后，我们对这个子集应用归一化变换，这样就只会对特征进行归一化，而不影响标签。

    batch_size = 50
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1), nn.BatchNorm1d(1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 通过调用net(X)生成预测并计算损失l（前向传播）
            trainer.zero_grad()
            l.backward()  # 通过进行反向传播来计算梯度  算出每个系数的偏导
            trainer.step()  # 通过调用优化器来更新模型参数  系数 = 系数 - 梯度*学习率  batchsize放在了损失函数里
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1},loss{l:f}')


# 工具模板
class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 工具模板
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 前面表示不是标量，后面表示某一维不是单个元素
        y_hat = y_hat.argmax(axis=1)  # axis=1 沿着第二个维度（列），找每个样本的最大值的索引
    cmp = y_hat.type(y.dtype) == y  # 索引比较，1说明预测对了
    return float(cmp.type(y.dtype).sum())


# 工具模板
def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # Accumulator里创建2个变量，分别为 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # 每个批量中，正确预测数、预测总数累加起来
    return metric[0] / metric[1]


# 工具模板
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 工具模板
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def softmax_classfi():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()  # 成交量做标准化
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    # 前80%做训练集，后20%做测试集
    data_iter_train = load_array((features[:len(features) * 0.8, :], labels[:len(features) * 0.8, :]), batch_size)
    data_iter_test = load_array((features[len(features) * 0.8:, :], labels[len(features) * 0.8:, :]), batch_size)

    # 初始化参数  每个输出都有一系列系数
    net = nn.Sequential(nn.Linear(2, 2))

    # net = nn.Sequential(nn.Linear(2, 1), nn.BatchNorm1d(1))  图像卷积的中间层才会用
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # y_hat是所有样本，y是所有样本正确类别的概率
    # 所以log的参数是所有样本正确类别的概率列表，然后log对列表里每个值求e为底的对数，还是返回张量
    # 交叉熵返回的就是每个样本正确类别预测概率的负对数值
    # 符号不用担心，对数的x小于1时，y是负数，前面有符号，变成正数，而这里log的参数是概率，他们永远在[0,1]，只是越靠近0，对数反而越大
    # 但熵要越小越好，为什么？
    # 熵值越小，数据越纯，分类越好。也就是log的参数越靠近1越好。也就是y_hat里正确类别的概率越大越好
    # 所以优化就变成了求交叉熵这个函数的最小值。还是通过梯度下降不停更新w和b求它的最小值

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    train_ch3(net, data_iter_train, data_iter_test, loss, num_epochs, trainer)

    # y_hat这2列分别代表 买，卖
    y_hat = torch.tensor([[0.3, 0.7], [0.2, 0.8]])  # 2个样本在2个类别的预测概率
    y = torch.tensor([0, 1])  # y通过马后炮观察得出，这代表从第一个样本开始：不动，买，不动，卖    ！！！！！做分类的人少可能就是标注数据难

    y_hat[[0, 1], y]  # 前面的中括号是要取哪个样本，y代表取每个样本的哪个列
    # 意思就是，y_hat第0个样本，正确答案是第0列，  y_hat第2个样本，正确答案是第0列
    # python里这样对应取值叫做 zip函数，同时遍历两个列表


def mlp_classfi():
    DataFrame = pd.read_csv("backtest_bar_600438_5.csv", encoding='gbk')
    features = torch.from_numpy(DataFrame.iloc[:-1, 4:].values).to(torch.float32)  # 两个参数，收盘价与成交量
    features[:, 1] = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()  # 成交量做标准化
    labels = torch.from_numpy(DataFrame.iloc[1:, 4:5].values).to(torch.float32)

    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 50
    # 前80%做训练集，后20%做测试集
    data_iter_train = load_array((features[:len(features) * 0.8, :], labels[:len(features) * 0.8, :]), batch_size)
    data_iter_test = load_array((features[len(features) * 0.8:, :], labels[len(features) * 0.8:, :]), batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(2, 32),
                        nn.ReLU(),
                        nn.Linear(32, 2))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);

    batch_size, lr, num_epochs = 32, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_ch3(net, data_iter_train, data_iter_test, loss, num_epochs, trainer)

    """

    # 2024 03 09补充，不论之前是三维，四维，还是更多维，用了Flatten，只能展成2维
    # net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

  
    这里再捋一遍思路
    线性回归的模型用的是linear
    分类用的是线性回归linear基础上的softmax
    线性回归损失函数是均方误差
    分类用交叉熵
    都是求最小值
    线性回归梯度下降用的sgd
    分类也是sgd
 
    -----------------------------------------------------------
    # 3维还是Flatten成2维，只是和p4的框架实现相比，这里感知机多加了一层
    # p4里说过，无论多少维，Flatten都是展成为2维
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),  # 第一层输出256，因为有256个单元
                        nn.ReLU(),  # 多的一层用的是relu的激活函数，
                        nn.Linear(256, 10))
    # 与p5手动实现net区别是，p5就一层，数据进去，relu一下，直接输出，所以训练时循环一次net就够了，
    # 这里是两个层，训练时走net函数时，X先展成2维，进nn.Linear做矩阵乘法得到输出，然后输出走relu得到另一个输出，这是第一层，
    # 然后这个relu输出走最后一个nn.Linear做矩阵乘法得到最终10个输出，第二层就是输出层
    """


"""
创新点：https://www.xiaohongshu.com/user/profile/58b6dc405e87e72284731915?xhsshare=WeixinSession&appuid=5988a9a250c4b438e2221ef2&apptime=1711622114


单变量每个样本，是挖掘时间上两个相邻元素的关系，那只要够长，就能挖掘出价格走势的关系 比如close组成250个bar
多变量的话，是多个特征的变化，对结果的影响，可能不如单变量效果好，但是，这多个特征直接必然要有关联，如果我用指标，也许效果会提升？
多变量  正常排，输入，隐藏层权重 ， 这样理解是对的，收藏的那篇单变量多变量解释也是这样
多变量，原本是每个特征 和权重 乘，
transformer encoder多个头的结果压在一起，给decoder，
decoder循环，训练时，decoder inputs是未来的真实值；预测时，inputs是decoder的output上次的预测值

批量大小大点也没关系吧
看是不是预测序列长度太短导致测试结果不准，把预测都加上，看实盘结果，改最后一条的收盘价
莫非跟当年nlp 一样，预测效果差还是因为训练的太少？现在的时间序列有多大参数？


四个方法，新模型a+b，新模型分钟级a+b，传统指标+分钟级模型二分法过滤，传统指标+多级别模型二分法过滤
接下来就是挨个试时间序列模型，先跑起来但预测价格为次，然后想办法改成辅助指标交易点的二分法。

现有的都是通用的时间序列预测，分解季节，趋势，我做股票，只用他趋势+指标行不行？

1、期权

当前价格100

预计5天后 会涨到110， 那我就买个5块，以100元（未来的低价）在5天后买，的期权，到了那天，真的超过110了，那我就行权，以100买入2手，马上以110价格卖出
																  如果没有，那就损失个保证金
																  期间我想止盈了，比如现在105，我可以卖给别人
预计5天后 会跌到90，  那我就买个5块，以100元（未来的高价）在5天后卖，的期权，到了那天，真跌了，我就在市场以90买个，马上以100卖给对方

所以，我可以只买10u，预测100u的波动。赢一次赚90u，输一次损失10u

合约在波动时有强制平仓风险，而期权不怕波动，损失固定，只要够小就可以
对冲，持有现货怕跌，买看跌期权，如果没跌，不行权，利润能盖过权利金。跌了，则对冲损失。  
现货1000u1份，跌20%到800u，损失200u，要想对冲200u，则买1张看跌期权 1000u在5天后卖给对方，盈利为200u-1张的权利金（大概1%，也就是10u），即190u，
也就是说，1%，就能对冲，而且波动了百分之多少，就相当于盈利了多少倍杠杆。
如果1份价格过高，1%也很贵，这样可以买千分之一，原来赚190，现在只能转19u了，但花了1u。

他们现在买卖的，都是转卖的。比如 预计5天后到90，现在95，说明盈利5块了，那我卖5块，继续跌我才有的赚，不赚钱我5块全亏。
别看探索里转售的，直接买看涨期权，看跌期权，选日期，选心目中的价格，买个0.1张或更少即可。

币安的期权不用操作，赚了自动利润打到账户，赔了权利金直接扣掉。

期权不是100倍杠杆，
相当于自己开了个特殊杠杆
权利金就是当前和预期的价差，把价差补上，相当于自己买了个多倍合约，拥有了一个币，盈利就是1个币的价格相对期权价格的波动
这个价差越靠近当前价格，越正常，越离谱就越便宜（发生概率低，大概率扔钱）
所以，远期小概率时间收益最高，将近10倍收益
近期就是个普通的1倍，或者说一个币的波动利润

长期看涨的人远期看空对冲， 比如3800刀买了1个eth，就开空220刀1个eth  用5.7%的代价，换一个月的安心，当然涨过5.7%这样算没动，不然还要亏个5.7%作为买放心的代价

如果长期看空，那就买远期看涨对冲，比如1900刀一倍开空，270开涨，7%的代价，
总之在行权那天，如果涨破3800,270期权能弥补开空损失，但怕行权日之前涨超1900导致爆仓后又回落
				如果跌破3800，合约赚钱，但利润要大于270刀，才是真的赚

除了对冲以外，期权相当于一个防插针的合约
比如 当前3800，5刀明天看涨，到4000，才赚15刀
所以期权也有问题，就是权利金价格不稳定，200刀的利润，权利金应该是2刀  结果2刀利润只有7刀，降低了30倍

-----------------------------------------
最初是用模型做策略。先看了书，挑了论文，结论发现两个问题：
1、论文做策略难度大，不好复现，AI生成的代码错了自己也不知道，一个方法涉及的引用，需要几个月从头学
2、目前的策略都不赚钱。想微调都没方向

后来书整个读完，实践完，发现策略都不行。

接下来进入自己的研究领域，先看的综述，找了能找的，发现都是 ANN，CNN，RNN，  former系的很少
在综述里发现，集成学习比单个好，如何至宝，   看了大量论文，发现难度高，提升小。  集成学习效果好是实话，但付出与收获不成正比。
如果量化给我带来的是瞎试，我不如把这时间花在别的方向
反正旧领域已经被玩烂了，现在只有不断用新模型去解决老问题。

我目前只是加了准确的点位判断，加了止损，加了多级别提醒
唯一缺一样，把明显不能买的买入信号过滤掉，  怎么过滤？如果贸然用策略写死，可能有很多条件判断不到，能写，但肯定会将条件卡的更严，这是下策
所以现在寄希望于特征提取解决过滤问题
如果这条路最后行不通，那我只能用新模型解决老问题，搞个A+B应付了事，从此不再考虑量化策略，不再投入时间，就现有策略加个下策写死，到此为止了。


我只想在现有交易信号触发时，判断是不是上级形态能不能买，做个过滤，输出分类

回归类的难度大，实际胜率低
实际需要的是标记出买卖点，人工标是个体力活（交易点出现后5天内涨超2%就算能买？），也没法创新
要不用强化学习的方式标？
一旦交易点标出，作为训练数据，那之前的回归模型就不能用这些了，得改个分类模型
那是不是直接强化学习和传统指标结合的方式更直接？


A股特殊，先看大盘，大盘跌久了，才能买各个标的
所以一年的机会并不多
目前高位刚回落，再等等吧
再次证明，没有日线背离的铺垫，或者处于上行趋势，那小级别背离就是没用
上次只是把提示功能优化了，本地读写仓位状态+止损，并没有优化策略
首先应该把多级别过滤加上，减少提示频率
或者不要多级别过滤？AI抽取上级特征  现在在试验这步
	想到个简单方法，AI不用分类，交易信号出现后，多级别直接回归，判断明天涨跌，决定能不能买？估计会不准
---------------------------------------
Trend Prediction in Finance
机器学习金融预测有两种：价格预测，交易点预测

应该是先有理论觉得可以，然后实验去验证。
比如觉得这个模型在这个领域有合理性，能提升，就去验证。
方法太多了，盲目去试不可取，还是要从合理性出发

我希望的位置信息，是判断特征总结出这一段探底之后就会出现反转，
所以时间段得长一点，包含了探底与反转，抓出反转点所在区间的特征，预测现在就是反转点的概率有多大，给我提示
这一对信号，探底与反转，的相对位置信息
自注意力，没有位置信息，一个办法是位置编码
就是把位置信息做成形状相同的矩阵加到输入的矩阵，分绝对和相对，我肯定用相对
关于我自己的初衷leaner:
我想要的是，比如日线处于底背离的形态，30分钟也是底背离形态，然后此时均线，KDJ，MACD等等指标可能具有其他我没挖掘出的特征，此时买入后大概率涨。
这些特征，比如只需要250个样本，很久以前的样本跟此刻的形态没什么关系（0524，不，15分钟级别，那就有关系了，因为省我看日线了），所以不需要transformer-based 模型，用CNN抽象出特征就行。
为什么不用LSTM？LSTM只是预测价格，而CNN

(把新的挖掘特征的方法，用到老的框架，进行预测
比如背离，破均线，均线走势等，他们常判断失效，但走势越明显，判断正确的概率就越大。  
现在的问题在于难以挖掘所谓的“走势非常明显”，如果挖掘出来，再把它喂给模型去抽取特征共性，再做判断，成功率会比传统方式高。
只要挖掘出特征，再用不同的模型，都有提升，就可以发不同的文章。)


描述借鉴
In this paper, we propose a novel ... is first applied to mine significant technical trading patterns from the technical indicators ...
we present StockFormer, a hybrid trading
machine that integrates the forward modeling capabilities of predictive coding with the advantages of
RL agents in policy fexibility

--------------------------------------------------------
什么是指标？传统指标源于对资金博弈的分析，是量价及背后势力的行为反应，有道理，后来参与的人多，好像锦上添花般成为了一种共识，真有人在指标出现交易信号时出手（包括无人操作的量化策略）。
什么是策略？狭义的策略是指单个或多个指标所提炼出的交易信号，广义的策略也包括大资金对市场的操作时的计谋。除了买卖点，还应该有仓位管理（止盈止损等）。
指标为什么会失效？但由于市场的价格是多方合力的结果，有庄家（团队大资金）、游资（个体大资金）、散户（个体小资金）三种群体多方势力彼此尔虞我诈，通过不同策略（或计谋）从其他人口袋里骗钱。如果只奉行单个策略，当某个策略的信众力量薄弱时，该策略就会失效【王立新那论文】。
既然会失效，那我们在市场中获胜的理论依据是什么？增加盈利概率。金融本质是社会未来发展需求导致的资金流动，长期的经济趋势决定了大方向，选对了方向本就会使我们盈利的概率大于50%，为了利益最大化，则需要看长做短，在短期（一两年、几个月、几天、几小时、几分钟）博弈中寻找当前时间窗口内胜率较大的交易点位，使盈利概率在50%基础上再增加一点。市场看似水很深，但别害怕，纵使市场复杂，可结果无非涨跌，增加胜率具体如何做？对于资金量无法大到可以操纵市场的交易者来说，盈利只能来自于跟踪大资金量，要么在某方势头强劲时加入他们，要么在某方强弩之末时加入另一方。如何判断当前某方势力强弱？根据指标。非市场操作者能做的无非是组合不同时间级别的不同指标，筛选出成功率最高的交易机会。
那什么样的指标组合成功率高呢？比如同样是高位回落的底背离，双顶情况下的失败率远高于单顶；日线背离情况下小级别背离时反转的概率远高于小级别背离。这些都是传统指标的经验。现在需要做的，是在传统技术指标（大部分指标大同小异，主流的代表是macd，kdj，ma）的基础上，抽取指标有效时所在时间窗口的所有特征记为1，抽取指标失效时的特征记为0，以此实现在大于50%的胜率基础上进一步增加盈利概率。
这个方法的意义：1、对于任何现有策略（传统指标策略，各种经典策略），该方法可以通过识别失效有效情况的特征增加策略胜率；2、在逻辑上具有良好的可解释性；3、相对于单独使用时间序列预测模型，他们要么是通过拟合直接预测价格（回归），由于每一刻的价格都要预测，胜率做不到太高，而我是传统指标出现时才需要预测一下，胜率更高，而且回归有个问题，预测价格确定了数值，但预测涨1.5%还是2.5%，难度很大，可能某个人心情好就多买了点，就会造成预测准确率下降，但其实这并没有关系，反正涨了，盈利了；要么模型先预测价格，然后不需要那么准确，只把预测价格的涨跌作为2分类。 但预测下一刻，几刻的方向，比较准确，他能预测是否未来几个月几天价格走势要彻底反转么？很蓝的啦，过于依赖模型，准确率会很快达到瓶颈难以提升，也降低了可解释性，但这么做的好处是训练数据简单。而用模型辅助抽取特征的方式，它不是在做实际预测，而是做特征匹配，唯一的难点就在于训练数据需要标注。
再捋一下，
时间序列也抽取了某个时间窗口的特征，看现在的价格走势满足过去哪些特征，以此特征做识别，识别以前遇到这种情况价格会涨还是跌。
我的方法也抽取了某个时间窗口的特征，看现在的价格走势满足过去哪些特征，以此特征做识别，识别以前遇到这种情况指标会不会失效。

理论上来说，我也像XGBoost和残差一样，在优化和100%概率之间的差距，只是我是通过过滤的方式

所以我的分类才是更好的分类，但以前没人这么搞，因为数据标注太难。难度在于：
1、数据量大，假如1万个股票，10年数据，需要一个个标注哪个交易点有效，哪个失效。  （我通过回测+代码过滤实现）
2、上面1说的是单个级别。实际的指标运用时，常常会用多级别组合，比如日线背离前提下，分钟线背离，而正常的模型都是一个级别一个模型，两个级别模型之间没关系。而我想实现的，是 
15分钟级别数据进来时，指标判断买卖点出现，此时模型回去抓取30分钟，60分钟，日线级别250个时间窗口的特征，综合判断当前15分钟指标会失效还是有效；
30分钟进来，会判断60，日线级别， 就是只找当前级别的上级。  抓当前级别特征用处不大，因为都知道自己背离了，何必多次一举，肯定抓上级。
所以训练时需要多级别同时， （我有策略框架，这没问题）
问题：
	30分钟级别有指标信号，一个模型去判断多个上级？
	或者，			  一个模型去判断一个上级，然后几个模型融合
	那这样每次的模型数量就不一致了。

那接下来的问题就是
	模型与多级别怎么搭配？
	要不要再加个和大盘的关系？  和大盘有关的行业股，必须看大盘（日线，只要不是跌势就行，上行与震荡都能买），必须看高级别（跟大盘要求一样）。
	选什么模型用于抽取固定时间窗口内的特征。  （好像CNN,RNN,transformer-based的改成2分类的都可以，注意不是预测价格之后的二分类）

	这个只能用收益率，和单个模型对比，和单个指标对比。
	我最后的试验结果用王立新那种回测作为结果









股 4h， 8 30m , 16 15m
B 24h,  48 30m

SegRNN2023 也是分段 + 通道独立

PatchTST2023 属于Transformers
TiDE2023 Dlinear2023 属于 MLPs
TimesNet2023 属于CNNs

我能不能也用简单的MLP做分类 

小红书展示换衣服，换发型的模型效果，git地址

要不要分类，怎么分类

画图对真实价格的预测
尝试单变量
尝试指标特征

**2022 Dlinear**
双线性结构。decomposition先分解原数据的趋势和季节。
论文里的金融预测图
对于周期性不强的金融数据，mse这类评价指标低真没啥用，图说明，模型只是简单延续了趋势，并不能预测趋势反转，
那我退一步，既然预测出日线趋势反转的，比如涨久了刚跌，跌久了刚涨
但这玩意我自己用均线做也能判断，传统方法只能叠加信号，背离+均线反转
那模型预测了个啥？代替均线的趋势？


从原理方面分析，为什么失败的多：
金融市场本质看预期，受新闻，多方势力影响。
新闻会短期改变力量，但一天或几天内会恢复。
单边，震荡
时间序列预测能够预测出趋势，能预测周期，但与其他时间序列不同的是，
金融是0和博弈，一方的利润来自于另一方的亏损，双方力量一定会变化，也就是趋势一定会变，但时间灵活，周期长短也灵活，
模型能看出趋势延续
能用最近的周期模仿未来一小段时间的周期
但是，250个样本的顶背离，是经验，是几十上百次背离得出的经验，甚至还有多次背离，
把所有股票，历史所有走势都给模型，他能总结出背离么
换言之，模型能像人类一样总结出各种指标对应的买卖规则么？甚至包括不一定有用的政策和新闻
难

既然无法全能，那就只让模型做一小部分事。

指标做指标的判断， 
空头趋势尽，连续底背离，均线调头向上，放量
此时，buy + fuzzy ，核心是预测出趋势反转
趋势中呢？ 我是全监控，所以不会有趋势中的出现

模型咋办？
金融中，大周期在小级别也是趋势的变化，不如我就只用趋势。 分段，通道独立，趋势，  模型只需要预测出趋势反转，也许模型只是比fuzzy的均线更灵敏而已


那换个方式，我告诉模型各种指标，告诉他有效的买卖点，数据集的质量是很大的工作量，而且，这不是成强化学习了？工程的应用，学术创新在哪里？
解决问题的方法：用了个新数据集+老模型 解决了个老问题


大道至简，我就用简单的cnn+传统指标。 再加个分段和通道独立？  通道先是几个价格，后期再换指标？
CNN主要判断现在是不是趋势要反转了，是，否。
比如顶背离刚下来，CNN判断出上升反转为下降了，此时指标提示买，也不买，指标提示卖，要卖
顶背离下来过程中震荡，或反弹，CNN判断并未改变趋势，有信号也不参与
直到底背离了，并且传统指标提示买入，这时CNN判断趋势反转了，可以买，过段时间，还在上升趋势内，提示卖，也不卖
所以CNN只存储了现在的状态是上升趋势，还是下降趋势。

训练CNN需要标注训练数据，但这个没关系，不同标的都可以
是否需要集成学习呢？ CNN应该是个弱学习器吧？可我只有一个CNN，传统指标或者fuzzy没有系数，咋办？
是否需要强化学习呢？ 市场的变化本质是博弈，新的变化影响因素太多，不一定是规律，模型学会了，可能会学偏，自己也没法知道他是不是学偏了，没法控制
"""

if __name__ == '__main__':
    linear_regression()
    linear_regression_batchnorm()
