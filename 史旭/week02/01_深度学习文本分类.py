import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 调整模型全链接网络层数和节点数，查看loss损失值的变化

# 1.读取数据
data = pd.read_csv("../data/dataset.csv", header=None, sep="\t")
texts = data[0]
labels = data[1]

# 2.提取特征向量
# labels
label_to_index = {label: index for index, label in enumerate(set(labels))}
labels_vector = [label_to_index.get(label, 0) for label in labels]
labels_tensor = torch.tensor(labels_vector)


# texts（封装）
class BowDataset(Dataset):
    def __init__(self, texts, labels_tensor):
        self.texts = texts
        self.labels_tensor = labels_tensor
        self.char_dict = self.get_char_dict(texts)
        self.texts_tensor = self.get_text_tensor(texts)

    # 词字典
    def get_char_dict(self, texts):
        char_dict = {"<nan>": 0}
        for text in texts:
            for char in text:
                if char not in char_dict:
                    char_dict[char] = len(char_dict)
        return char_dict

    # 获取词频向量
    def get_text_tensor(self, texts):
        texts_vector = []
        for text in texts:
            text_tensor = torch.zeros(len(self.char_dict))
            for char in text:
                if self.char_dict.get(char, 0) != 0:
                    text_tensor[self.char_dict.get(char)] += 1
            texts_vector.append(text_tensor)

        return torch.stack(texts_vector)

    # 被DataLoader调用时拦截处理
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (self.texts_tensor[index], self.labels_tensor[index])


# 3.模型
class TorchModule(torch.nn.Module):
    def __init__(self, input_dim, out_dim, *hidden_dim):
        super(TorchModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.network = self.get_network(input_dim, out_dim, *hidden_dim)

    # 根据 传入的隐藏层节点数，创建全链接网络
    def get_network(self, input_dim, out_dim, *hidden_dim):
        layers = []
        in_features = input_dim
        for dim in hidden_dim:
            layers.append(torch.nn.Linear(in_features, dim))
            layers.append(torch.nn.ReLU())

            # 维度更新
            in_features = dim
        layers.append(torch.nn.Linear(in_features, out_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 4.实例化（数据，模型，损失函数，优化器）
bow_dataset = BowDataset(texts, labels_tensor)
dataloader = DataLoader(bow_dataset, batch_size=32, shuffle=True)

# 多个 不同层数不同节点的模型
input_dim = len(bow_dataset.char_dict)
out_dim = len(label_to_index)
hidden_dim1 = (128,)
hidden_dim2 = (128, 50)
# hidden_dim3 = (250, 128)
hidden_dim3 = (250,)
hidden_dim4 = (30,)
hidden_dim5 = (30, 15)
torchmodule1 = TorchModule(input_dim, out_dim, *hidden_dim1)
torchmodule2 = TorchModule(input_dim, out_dim, *hidden_dim2)
torchmodule3 = TorchModule(input_dim, out_dim, *hidden_dim3)
torchmodule4 = TorchModule(input_dim, out_dim, *hidden_dim4)
torchmodule5 = TorchModule(input_dim, out_dim, *hidden_dim5)

# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 优化器
optimizer1 = torch.optim.SGD(torchmodule1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(torchmodule2.parameters(), lr=0.01)
optimizer3 = torch.optim.SGD(torchmodule3.parameters(), lr=0.01)
optimizer4 = torch.optim.SGD(torchmodule4.parameters(), lr=0.01)
optimizer5 = torch.optim.SGD(torchmodule5.parameters(), lr=0.01)


# 5.模型训练
def model_loss_optimizer(torchmodule, loss_func, input_tensor, label_tensor, optimizer):
    # 模型计算
    out_tensor = torchmodule.forward(input_tensor)
    loss = loss_func(out_tensor, label_tensor)
    # 梯度清零，反向计算梯度，调参
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def modle_train(torchmodule1, torchmodule2, torchmodule3, torchmodule4, torchmodule5,
                dataloader, loss_func, optimizer1, optimizer2, optimizer3, optimizer4, optimizer5):
    lossmean_list1, lossmean_list2, lossmean_list3, lossmean_list4, lossmean_list5 = [], [], [], [], []

    # 训练模式
    torchmodule1.train()
    torchmodule2.train()
    torchmodule3.train()
    torchmodule4.train()
    torchmodule5.train()

    # 循环训练
    for epoch in range(10):
        lossmean1, lossmean2, lossmean3, lossmean4, lossmean5 = 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0

        # 分批次训练
        for index, (input_tensor, label_tensor) in enumerate(dataloader):
            # 模型计算
            loss1 = model_loss_optimizer(torchmodule1, loss_func, input_tensor, label_tensor, optimizer1)
            loss2 = model_loss_optimizer(torchmodule2, loss_func, input_tensor, label_tensor, optimizer2)
            loss3 = model_loss_optimizer(torchmodule3, loss_func, input_tensor, label_tensor, optimizer3)
            loss4 = model_loss_optimizer(torchmodule4, loss_func, input_tensor, label_tensor, optimizer4)
            loss5 = model_loss_optimizer(torchmodule5, loss_func, input_tensor, label_tensor, optimizer5)

            # 用来对比模型差异
            lossmean1 += loss1.item()
            lossmean2 += loss2.item()
            lossmean3 += loss3.item()
            lossmean4 += loss4.item()
            lossmean5 += loss5.item()
            count += 1

            if index % 50 == 0:
                print(f"模型隐藏层结构：{torchmodule1.hidden_dim}，第{epoch + 1}次训练，第{index}批次，loss：{loss1}")
                print(f"模型隐藏层结构：{torchmodule2.hidden_dim}，第{epoch + 1}次训练，第{index}批次，loss：{loss2}")
                print(f"模型隐藏层结构：{torchmodule3.hidden_dim}，第{epoch + 1}次训练，第{index}批次，loss：{loss3}")
                print(f"模型隐藏层结构：{torchmodule4.hidden_dim}，第{epoch + 1}次训练，第{index}批次，loss：{loss4}")
                print(f"模型隐藏层结构：{torchmodule5.hidden_dim}，第{epoch + 1}次训练，第{index}批次，loss：{loss5}")
                print("-" * 100)

        # 用来对比模型差异
        lossmean1 /= count
        lossmean2 /= count
        lossmean3 /= count
        lossmean4 /= count
        lossmean5 /= count
        print(f"训练第{epoch + 1}，模型隐藏层结构：{torchmodule1.hidden_dim}，loss均值：{lossmean1}")
        print(f"训练第{epoch + 1}，模型隐藏层结构：{torchmodule2.hidden_dim}，loss均值：{lossmean2}")
        print(f"训练第{epoch + 1}，模型隐藏层结构：{torchmodule3.hidden_dim}，loss均值：{lossmean3}")
        print(f"训练第{epoch + 1}，模型隐藏层结构：{torchmodule4.hidden_dim}，loss均值：{lossmean4}")
        print(f"训练第{epoch + 1}，模型隐藏层结构：{torchmodule5.hidden_dim}，loss均值：{lossmean5}")
        print("=" * 100)

        lossmean_list1.append(lossmean1)
        lossmean_list2.append(lossmean2)
        lossmean_list3.append(lossmean3)
        lossmean_list4.append(lossmean4)
        lossmean_list5.append(lossmean5)

    # 画图 展示模型之间的差异
    plt.figure(figsize=(12, 4))
    plt.plot(lossmean_list1, label=f"{torchmodule1.hidden_dim}", color="blue")
    plt.plot(lossmean_list2, label=f"{torchmodule2.hidden_dim}", color="red")
    plt.plot(lossmean_list3, label=f"{torchmodule3.hidden_dim}", color="black")
    plt.plot(lossmean_list4, label=f"{torchmodule4.hidden_dim}", color="orange")
    plt.plot(lossmean_list5, label=f"{torchmodule5.hidden_dim}", color="green")
    plt.xlabel("epoch")
    plt.ylabel("lossmean")
    plt.legend()
    plt.show()


# 6.获取不同模型训练后的结果
modle_train(torchmodule1, torchmodule2, torchmodule3, torchmodule4, torchmodule5,
            dataloader, loss_func, optimizer1, optimizer2, optimizer3, optimizer4, optimizer5)


# modle_train(torchmodule2, dataloader, loss_func, optimizer2)
# modle_train(torchmodule3, dataloader, loss_func, optimizer3)


# 7.预测方法
def pred_func(new_text, char_dict, index_to_label, input_dim, *torchmodule):
    # 提取预测文本 特征向量
    text_tensor = torch.zeros(input_dim)
    for char in new_text:
        if char_dict.get(char, 0) != 0:
            text_tensor[char_dict[char]] += 1

    # 输入张量 调整维度
    text_tensor = text_tensor.unsqueeze(dim=0)

    # 模型评估模式
    for model in torchmodule:
        model.eval()
        with torch.no_grad():
            pred_tensor = model.forward(text_tensor)
        # 获取 预测结果
        pred_maxvalue_index = pred_tensor.argmax(dim=1)
        pred_result = index_to_label[pred_maxvalue_index.item()]

        print(f"模型隐藏层结构：{model.hidden_dim}，输入 '{new_text}' 预测为: '{pred_result}'")


# 8.预测
# 获取 索引对应分类结果 dict字典
index_to_labels = {index: label for label, index in label_to_index.items()}

new_text = "帮我导航到北京"
pred_func(new_text, bow_dataset.char_dict, index_to_labels, input_dim,
          torchmodule1, torchmodule2, torchmodule3, torchmodule4, torchmodule5)

print("-" * 100)

new_text_2 = "查询明天北京的天气"
pred_func(new_text_2, bow_dataset.char_dict, index_to_labels, input_dim,
          torchmodule1, torchmodule2, torchmodule3, torchmodule4, torchmodule5)

# 结论：（与原来一层网络比较）
# 当第一层节点数不变，而层数增加时：损失值开始阶段下降缓慢，中间阶段下降迅速，末尾阶段趋于平缓，整个阶段未曾低于原有层数
# 当层数和节点数都增加时：损失值开始阶段下降缓慢，中间阶段下降迅速，末尾阶段趋于平缓，整个阶段未曾低于原有层数

# 当层数不变，节点数增加时：损失值下降曲线与原来几乎一致
# 当层数不变，节点数下降时：损失值下降曲线与原来几乎一致
