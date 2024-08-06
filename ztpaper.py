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
A股，只能从沪深300里找上市后低位震荡未拉升的，基本面过得去的，未来不是绝路的股票，做长线波段。能管两年
拉过的，不会再拉，庄家知道里面散户多。
没拉过的，一般都会有三年以上收集筹码的时间。这个题材涨了，他没涨，因为庄家还在收集筹码，这个股票以后一定有机会。

1、取股票数据
2、运行回测
3、交易点时间及价格 保存到交易csv文件
4、读取股票数据，读取交易csv文件
5、遍历交易csv文件的每行数据，找到对应股票数据的日期，推3、5、10天？如果最高价大于买入价5%，就算合理   跌同理
6、将 合理的交易日期，保存到新文件
7、组织训练数据，多变量时间序列分类，变长，2类（买卖）
8、timesnet到底是压缩，还是每个特征单独训练？  用timesnet进行分类训练

量化公司，代写策略，部署策略

timesnet所用框架的处理流程：
	1、创建分类试验对象，与回归区别是，初始化对象时会自动取特征数做enc_in，自动获取分类数量还，此时虽调用了加载数据，但没传参只是创建了个空对象
	2、调用train()训练
	3、训练函数入口，加载数据，调用data_provider，选择传入的time series数据类型，比如UEA，就用对应类型的解析工具，把ts文件转为df
	4、加载完数据，开始epoch循环，此时调用了self.model.train()，但这里没传参，应该是调父类的方法
	5、循环训练数据，把数据给self.model调forward。

	            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

现在导师的任务是，一方面按原思路继续做，另一方面找相关论文，金融分类的

https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html
日语，12个特征，9个分类，
每一行是把每个特征的所有样本值组成一个向量。 
以第一行数据为例， 这是20个样本的数据，  特征1的20个样本的值放一起，特征2的20个放一起，以:间隔特征，最后一个值是分类的标签
第二行数据是26个样本的数据。12个特征

这种组织方式跟具体怎么用无关，训练时可以压缩用，也可以每个特征单独训练，只是这是一种数据组织格式

固定的方式比如Uwave  3个特征，每行3个:，每个特征有315个样本的数据。

我的股票数据，有开高低收量，5个特征（其实时间列不要也行，因为价格跟几点，几号，没啥关系）  ，
我一行数据，变长，是130个bar，就是130个样本，才决定买卖， 那这行就5个:，第一段130个开的值，一个:，然后130个高，一个：，最后是0，代表卖
也就是130条数据，是上一次买卖点，到这次买卖点之间的长度
那这一个股票，比如上市以来就100个买卖点，那就是100行数据。
那这数据够么？我看日语也就260多行
要想多训练点数据，多个股票是否可以训练同一个模型呢？

另一种是定长，我就用300个bar，130个bar就卖了，我用0或均值去填充依次填充5个特征向量的其他170个空值，这样岂不是很影响结果么？

以上就是数据组成方式，跟数据使用方式无关
至于我拿到数据之后，怎么训练，用压缩，还是特征单独，另说
------------------------------------------------------
1、标注数据

标注数据
要不用强化学习的方式标？
当初写错的模糊策略，也能当作标注数据集的工具
按照各种策略先回测，把回测的数据，买入后还跌的剔除掉，买入后没涨的剔除掉，这样过滤一下，就拿到优质数据了。
要么每次2%也行，2%代表当年活期年利率  5天内最高价超过买入价2%就算有效买点？
个股加换手
创新点：https://www.xiaohongshu.com/user/profile/58b6dc405e87e72284731915?xhsshare=WeixinSession&appuid=5988a9a250c4b438e2221ef2&apptime=1711622114



机器学习金融预测有两种：价格预测，交易点预测

我希望的位置信息，是判断特征总结出这一段探底之后就会出现反转，
所以时间段得长一点，包含了探底与反转，抓出反转点所在区间的特征，预测现在就是反转点的概率有多大，给我提示
这一对信号，探底与反转，的相对位置信息
自注意力，没有位置信息，一个办法是位置编码
就是把位置信息做成形状相同的矩阵加到输入的矩阵，分绝对和相对，我肯定用相对

关于我自己的初衷leaner:
我想要的是，比如日线处于底背离的形态，30分钟也是底背离形态，然后此时均线，KDJ，MACD等等指标可能具有其他我没挖掘出的特征，此时买入后大概率涨。
这些特征，比如只需要250个样本，很久以前的样本跟此刻的形态没什么关系
（0524，不，15分钟级别，那就有关系了，因为省我看日线了），所以不需要transformer-based 模型，用CNN抽象出特征就行。
为什么不用LSTM？LSTM只是预测价格

(把新的挖掘特征的方法，用到老的框架，进行预测
比如背离，破均线，均线走势等，他们常判断失效，但走势越明显，判断正确的概率就越大。  
现在的问题在于难以挖掘所谓的“走势非常明显”，如果挖掘出来，再把它喂给模型去抽取特征共性，再做判断，成功率会比传统方式高。
只要挖掘出特征，再用不同的模型，都有提升，就可以发不同的文章。)


把明显不能买的买入信号过滤掉，特征提取解决过滤问题

我只想在现有交易信号触发时，判断是不是上级形态能不能买，做个过滤，输出分类

再次证明，没有日线背离的铺垫，或者处于上行趋势，那小级别背离就是没用

首先应该把多级别过滤加上，减少提示频率
或者不要多级别过滤？AI抽取上级特征  现在在试验这步

想到个简单方法，AI不用分类，交易信号出现后，多级别直接回归，判断明天涨跌，决定能不能买？估计会不准



那换个方式，我告诉模型各种指标，告诉他有效的买卖点，数据集的质量是很大的工作量，而且，这不是成强化学习了？工程的应用，学术创新在哪里？
解决问题的方法：用了个新数据集+老模型 解决了个老问题

大道至简，我就用简单的cnn+传统指标。 
CNN主要判断现在是不是趋势要反转了，是，否。
比如顶背离刚下来，CNN判断出上升反转为下降了，此时指标提示买，也不买，指标提示卖，要卖
顶背离下来过程中震荡，或反弹，CNN判断并未改变趋势，有信号也不参与
直到底背离了，并且传统指标提示买入，这时CNN判断趋势反转了，可以买，过段时间，还在上升趋势内，提示卖，也不卖
所以CNN只存储了现在的状态是上升趋势，还是下降趋势。

训练CNN需要标注训练数据，但这个没关系，不同标的都可以
是否需要集成学习呢？ CNN应该是个弱学习器吧？可我只有一个CNN，传统指标或者fuzzy没有系数，咋办？
是否需要强化学习呢？ 市场的变化本质是博弈，新的变化影响因素太多，不一定是规律，模型学会了，可能会学偏，自己也没法知道他是不是学偏了，没法控制


--------------------------------------------------------
2、用标注好的数据训练模型


什么是指标？传统指标源于对资金博弈的分析，是量价及背后势力的行为反应，有道理，后来参与的人多成为共识，真有人在指标出现交易信号时出手（包括无人操作的量化策略）。
什么是策略？狭义的策略是指单个或多个指标所提炼出的交易信号，广义的策略也包括大资金对市场的操作时的计谋。除了买卖点，还应该有仓位管理（止盈止损等）。
指标为什么会失效？但由于市场的价格是多方合力的结果，有庄家（团队大资金）、游资（个体大资金）、散户（个体小资金）三种群体多方势力彼此尔虞我诈，通过不同策略（或计谋）从其他人口袋里骗钱。如果只奉行单个策略，当某个策略的信众力量薄弱时，该策略就会失效【王立新那论文】。
既然会失效，那我们在市场中获胜的理论依据是什么？增加盈利概率。金融本质是社会未来发展需求导致的资金流动，长期的经济趋势决定了大方向，选对了方向本就会使我们盈利的概率大于50%，为了利益最大化，则需要看长做短，在短期（一两年、几个月、几天、几小时、几分钟）博弈中寻找当前时间窗口内胜率较大的交易点位，使盈利概率在50%基础上再增加一点。市场看似水很深，但别害怕，纵使市场复杂，可结果无非涨跌，增加胜率具体如何做？对于资金量无法大到可以操纵市场的交易者来说，盈利只能来自于跟踪大资金量，要么在某方势头强劲时加入他们，要么在某方强弩之末时加入另一方。如何判断当前某方势力强弱？根据指标。非市场操作者能做的无非是组合不同时间级别的不同指标，筛选出成功率最高的交易机会。
那什么样的指标组合成功率高呢？比如同样是高位回落的底背离，双顶情况下的失败率远高于单顶；日线背离情况下小级别背离时反转的概率远高于小级别背离。这些都是传统指标的经验。现在需要做的，是在传统技术指标（大部分指标大同小异，主流的代表是macd，kdj，ma）的基础上，抽取指标有效时所在时间窗口的所有特征记为1，抽取指标失效时的特征记为0，以此实现在大于50%的胜率基础上进一步增加盈利概率。
这个方法的意义：
1、对于任何现有策略（传统指标策略，各种经典策略），该方法可以通过识别失效有效情况的特征增加策略胜率；
2、在逻辑上具有良好的可解释性；
3、相对于单独使用时间序列预测模型，他们要么是通过拟合直接预测价格（回归），由于每一刻的价格都要预测，胜率做不到太高，而我是传统指标出现时才需要预测一下，胜率更高，而且回归有个问题，预测价格确定了数值，但预测涨1.5%还是2.5%，难度很大，可能某个人心情好就多买了点，就会造成预测准确率下降，但其实这并没有关系，反正涨了，盈利了；要么模型先预测价格，然后不需要那么准确，只把预测价格的涨跌作为2分类。 但预测下一刻，几刻的方向，比较准确，他能预测是否未来几个月几天价格走势要彻底反转么？很蓝的啦，过于依赖模型，准确率会很快达到瓶颈难以提升，也降低了可解释性，但这么做的好处是训练数据简单。而用模型辅助抽取特征的方式，它不是在做实际预测，而是做特征匹配，唯一的难点就在于训练数据需要标注。

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
	或者，一个模型去判断一个上级，然后几个模型融合
	那这样每次的模型数量就不一致了。
	模型与多级别怎么搭配？
	要不要再加个和大盘的关系？  和大盘有关的行业股，必须看大盘（日线，只要不是跌势就行，上行与震荡都能买），必须看高级别（跟大盘要求一样）。
	选什么模型用于抽取固定时间窗口内的特征。  （好像CNN,RNN,transformer-based的改成2分类的都可以，注意不是预测价格之后的二分类）
	这个只能用收益率做评价指标，和单个模型对比，和单个指标对比。
	我最后的试验结果用王立新那种回测作为结果

---------------------------------------------------------
3、用期权进行策略实盘试验


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
"""

"""
# ztpaper

2017 transformer
transformer encoder多个头的结果压在一起，给decoder，
decoder循环，训练时，decoder inputs是未来的真实值；预测时，inputs是decoder的output上次的预测值

2020 ADD 股票数据特征提取
1、非纠缠结构分离 excess and market information
2、通过自蒸馏抹去部分因素
3、通过策略增加训练样本

**2021 informer**
稀疏概率降低复杂度，蒸馏使输入变长

**2021 Autoformer**
用到了decomposition，这个拆分（快速傅里叶变换FFT）基于 STL（季节性，趋势性），数据=趋势性数据+季节性数据（周期）+余项
auto-correlation代替注意力，因为作者认为数据不能简单由数值来判断，而应该根据趋势来判断，series-decomp用来把原始数据分解成季节（频繁小震荡）和趋势（大趋势），
auto-correlation利用随机过程的马尔可夫过程，以涛为时间窗口，计算分解后的两个周期对应那两小段的相关性，是个标量（最后加个softmax保证计算出的多个标量和是1），多个计算的标量连起来就是个趋势。
编码器是原始输入，解码器是拆分后的季节性和趋势性
结论：他对季节性周期性强的数据，预测效果最好，比如车流（有早晚高峰，工作日比周末车少）

**2022 Dlinear**
双线性结构。decomposition先分解原数据的趋势和季节。
论文里的金融预测图
对于周期性不强的金融数据，mse这类评价指标低真没啥用，图说明，模型只是简单延续了趋势，并不能预测趋势反转，
属于 MLPs

**2022 TDformer**
Trend Decomposition Transformer
20引用
各个趋势用自己适合的注意力机制

**2023 mamba**
循环神经网络 训练慢，推理快， transformer训练快，推理慢，mamba两个都快，训练时并行计算，推理时递归计算。
让SSM由时不变改为时变。速度快了2.8倍，内存少了86%

**2023 GBT**
Two-stage transformer framework for non-stationary time series forecasting
认为decoder inputs存在问题：不同局部窗口数据分布不均
现有的decoder inputs设计：
	informer：start token前移几位作为decoder的输入
	FEDFormer：趋势分解方式
这两个都没有解决过拟合问题
创新点：
	two stage:第一阶段通过对比学习获得输入序列的通用特征映射，第二阶段通过第一阶段和回归器输出的表征获得预测结果
	good begin：第一阶段的输出做第二阶段的输入

**2023 ITRANSFORMER**
INVERTED TRANSFORMERS AREEFFECTIVE FOR TIME SERIES FORECASTING
认为transformer-based模型不如Dlinear是因为token的不恰当使用
多个特征取了相同时间位置的token，但应该每个特征看作一个token
通过自注意力机制捕获特征之间的相关性
学习序列和全局的关系
金融预测效果比dlinear差一点

**2023 TimesNet**
时间序列的多周期性，互相重叠，互相影响
通过建模同时表示周期内和周期间的变化
autoformer团队的
属于CNNs

**2023 PatchTST**
原来的transformer是点对应token，现在是时间段对应token
通道独立，每个特征各自进transformer
目前效果最好的时间序列预测

2023SegRNN 
也是分段 + 通道独立

2023 StockFormer
Learning Hybrid Trading Machines with Predictive Coding
强化学习，但又跟普通的马科夫不一样，收益从0.4%提升到2%，一般
StockFormer consists of two training phases: predictive coding and policy learning

2023 Stock and market index prediction using Informer network
LSTM在时序预测上处于统治地位 ，Transformer有超越他的潜力，但现在这方面研究不够深入 
在长序列预测上，Informer比LSTM、Transformer表现更好

**2024 TimeMixer**
autoformer团队的
完全基于MLP模型，简单
多尺度混合：
	历史信息提取阶段：分解多个尺度变化
	预测阶段：多预测器混合多尺度信息
实验结果显示他是第一，TimesNet第二，PatchTST第三

**2024 TimesFM**
A DECODER-ONLY FOUNDATION MODEL FOR TIME-SERIES FORECASTING
zero-shot，不用在新数据集上训练，直接能用
预训练模型：patch input , decoder only , attention model
预训练数据集：作者做了个大型时序语料库
预测数据集：对不同领域可以zero-shot预测
NLP有跨任务zero-shot/few-shot learning的能力，这启发了时间序列预测，目前有两个研究方向：先预训练模型再给时间序列用；修改LLM模型给时间序列用
效果跟patchTST差不多

2024 MambaOut Do We Really Need Mamba for Vision
mamba适合长序列预测，视觉方面比不过卷积类模型

2024 Can Transformers Transform Financial Forecasting?
TFT、Informer、Autoformer、PatchTST 处理复杂时间序列数据很厉害，但之前有人用金融数据测的不严格，所以这次作者严格评测一下这四个。
作者之所以选这四个是因为他们在Nixtla这个神经网络实验评测的python库里
Q是当前对市场的分析，K是市场历史事件，V是市场历史事件对应当时的数据。
多头注意力类似同时考虑价格、成交量等
数据from July 1, 2007, to June 30, 2021.  Standard & Poor’s 100 (S&P 100) index
指标RMSE,MAE,MAPE
结论，Informer（长短期都强）最好，Autoformer（更适合短期）、PatchTST（更适合长期）一般，TFT不行

2024 Time Evidence Fusion Network
Evidence theory 1976年提出，融合信息做决策的 ，2000年有人用这个扩展为网络做分类，现在作者用这个做回归 Time Evidence Fusion Network
TEFN，把时间序列分为 时间BPA、通道BPA， 每个BPA又各自把序列扩散成多个mass,  (BPA源自模糊集理论)
本文说自己第一，PatchTST第二，TimesNet第三，autoformer和dlinear最后

**2024 KAN** 
Kolmogorov–Arnold Networks
能抽象出公式
MLP受universal approximation theorem启发，固定了激活函数作为层中的节点，
KAN受Kolmogorov-Arnold representation theorem启发，使用可学习的激活函数作为计算节点值的权重，因此没有权重矩阵，用1元函数代替了
KAN外表是MLP结构，内部是spline，使他不仅能学习特征，还能更精确。
KA是：如果f是个多元连续有界函数，则f可由有限个1元变量累加表示。人们想用ML学出这个1元函数，但这些函数可能是不可导的，所以KA被ML放弃了
但作者觉得科学中和生活中大部分函数都是可导的，应该关注典型的例子，而不是最坏的例子。
KAN的架构就是把每个1元函数参数化为B-spline curve，
例子，Alice想用KAN拟合2.22那个函数（两个输入，一个输出）。步骤是：
先建立（2,5,1）形状的全连接层（这是2层网络（x,x,x,x）这是3层网络），通过稀疏化训练，裁减掉80%隐藏层，得到（2,1,1）形状，假设出符号函数作为网络中的激活函数，继续训练直到准确率下降以确定参数，得到最终的符号函数。
这篇文章并不是独创KAN网络[n, 2n + 1, 1]，而是把这个网络泛化成任意的广度和深度。
We would also like to apply KANs to machinelearning-related tasks, which would require integrating KANs into current architectures, e.g., transformers – one may propose “kansformers” which replace MLPs by KANs in transformers.
KAN优点是可解释性和准确率，缺点是比MLP训练慢10倍，但作者认为这是个工程问题。
https://github.com/kindxiaoming/pykan
https://github.com/remigenet/SigKAN  预测均值或中位数让数据更平稳。

**2024 Diffusion-TS**

2024 A Novel Decision Ensemble Framework
Customized Attention-BiLSTM and XGBoost for Speculative Stock Price Forecasting
用BTC-USDT训练

2024 Literature Review
Machine Learning in Stock Predictions
prediction：Linear Regression, XGBoost, LSTM, ARIMA, and GARCH 这5个是预测价格
classification：Random Forest, Logistic Regression, Adaboost, GRU, and CNN 这5个分类趋势

2024 FinRobot 
An Open-Source AI Agent Platform
应用的结果不准。 用LLM分析新闻预测涨跌的
"""