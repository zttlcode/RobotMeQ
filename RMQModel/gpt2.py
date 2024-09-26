下面是使用 PyTorch 实现的 CNN 模型来处理股票的序列数据，解决二分类问题的代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 设置设备（如果有GPU则使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类，将序列数据处理为PyTorch可以使用的格式
class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# CNN模型定义
class StockCNN(nn.Module):
    def __init__(self, input_length, num_classes=1):
        super(StockCNN, self).__init__()
        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        # 最大池化层
        self.pool = nn.AdaptiveMaxPool1d(1)
        # 全连接层
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, num_classes)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)  # 池化并压缩维度
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 数据预处理函数
def preprocess_data(data, max_sequence_length):
    sequences = []
    labels = []
    for row in data:
        sequence_str, label_str = row.split(':')
        sequence = list(map(float, sequence_str.split(',')))
        label = int(label_str)
        sequences.append(sequence)
        labels.append(label)

    # 填充序列到固定长度
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', dtype='float32')
    return np.expand_dims(padded_sequences, 1), np.array(labels)

# 数据集准备
data = [
    "1.1,2.3,3.5,4.6:1",
    "2.2,3.4,1.5,0.6:0",
    "3.3,2.1,4.6:1",
    "1.1,3.3:0",
    # 可以加入更多样本...
]
max_sequence_length = 500
X, y = preprocess_data(data, max_sequence_length)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据加载
train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 模型、损失函数、优化器
model = StockCNN(input_length=max_sequence_length).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
        outputs = model(sequences)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test data: {100 * correct / total:.2f}%')
```

### 代码说明：

1. **StockDataset 类**：用于将输入数据转换为 PyTorch 的 `Dataset` 格式，以便通过 `DataLoader` 进行批处理和训练。

2. **StockCNN 模型**：
   - 使用 `Conv1d` 层进行一维卷积操作。
   - `AdaptiveMaxPool1d` 用于池化操作，将序列长度降低到1。
   - 全连接层 `fc1` 和 `fc2` 进行特征提取和最终的二分类输出，最后使用 `sigmoid` 激活函数来进行0或1的二分类。

3. **数据处理**：使用 `pad_sequences` 函数将不定长的序列填充为固定长度（500）。输入数据必须扩展维度，保证能够作为 CNN 输入。

4. **训练与测试**：通过定义 `DataLoader` 来进行批量数据的训练和测试，输出训练的损失值以及最终的测试集准确率。

你可以根据自己的需求来调整模型结构和超参数，例如卷积核数量、学习率、批大小等。