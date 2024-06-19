# ztpaper
1、准备三个数据集
kaggle，证券宝等，用国内数据和数字币
three public datasets from NASDAQ and
Chinese stock markets as well as the cryptocurrency market_
纳指，上指，BTC  日线  数据截止今天
分类数据集：按照各种策略先回测，把回测的数据，买入后还跌的剔除掉，买入后没涨的剔除掉，这样过滤一下，就拿到优质数据了。
要么每次2%也行，2%代表当年活期年利率  5天内最高价超过买入价2%就算有效买点？
2、旧策略回测，再看哪个分钟级别合适
如果单级别运行，还需要回测更多，来决定用哪个
如果按各个级别操作的话，15分钟比较合适
15分钟，1天16个bar，720个bar大概是2个月  日线一个背离都不算，15分钟短了
3、回测多个模型，大部分时序模型放到清华那个库里测，只是改一下数据集就好
2021 informer
2021 Autoformer
2022 Dlinear
2022 TDformer
2023 mamba  内存需要48g
2023 GBT https://github.com/origamisl/gbt
2023 ITRANSFORMER 
2023 TimesNet  https://github.com/thuml/TimesNet  内存拉满跑不起来
2023 PatchTST 
2024 TimeMixer  https://github.com/kwuking/TimeMixer  代码报错
2024 TimesFM https://github.com/google-research/timesfm
2024 KAN
2024 Diffusion-TS https://github.com/Y-debug-sys/Diffusion-TS
4、做个表格统计测试结果
5、预测价格 和 真实价格做个图
6、试试单变量，只有close，250个
7、评测模型预测股票和实盘这事做了

收藏视频
收藏书签
清理github
--------------------------------------------------

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
那我退一步，既然预测出日线趋势反转的，比如涨久了刚跌，跌久了刚涨
但这玩意我自己用均线做也能判断，传统方法只能叠加信号，背离+均线反转
那模型预测了个啥？代替均线的趋势？

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

**2023 PatchTST**
原来的transformer是点对应token，现在是时间段对应token
通道独立，每个特征各自进transformer
目前效果最好的时间序列预测

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
跟patchTST差不多

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
