import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

dataset=pd.read_csv('./dataset.csv',sep="\t",header=None)
texts=dataset[0].tolist()
string_labels=dataset[1].tolist()

label_to_index={label:i for i,label in enumerate(set(string_labels))}
numerical_labels=[label_to_index[label] for label in string_labels]

char_to_index={'<pad>':0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char]=len(char_to_index)

vocab_size=len(char_to_index)
max_len=40

class CharGRUDataset(Dataset):
    def __init__(self,texts,labels,char_to_index,max_len):
        self.texts=texts
        self.labels=torch.tensor(labels,dtype=torch.long)
        self.char_to_index=char_to_index
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text=self.texts[idx]
        indices=[self.char_to_index.get(char,0) for char in text[:self.max_len]]
        indices+=[0]*(self.max_len-len(indices))
        return torch.tensor(indices,dtype=torch.long),self.labels[idx]

class GRUClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,num_layers,dropout=0.2):
        super(GRUClassifier,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.gru=nn.GRU(embedding_dim,hidden_dim,num_layers=num_layers,batch_first=True,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_dim,output_dim)
        self.num_layers=num_layers

    def forward(self, x):
        embedded=self.embedding(x)
        gru_out,hidden=self.gru(embedded)

        if self.num_layers>1:
            last_hidden=hidden[-1]
        else:
            last_hidden=hidden.squeeze(0)
        out=self.dropout(last_hidden)
        out=self.fc(out)

        return out


gru_dataset=CharGRUDataset(texts,numerical_labels,char_to_index,max_len)
dataloader=DataLoader(gru_dataset,batch_size=32,shuffle=True)


embedding_dim=64
hidden_dim=128
output_dim=len(label_to_index)
gru_model=GRUClassifier(vocab_size,embedding_dim,hidden_dim,output_dim,num_layers=2,dropout=0.3)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(gru_model.parameters(),lr=0.001)


num_epochs=4

for epoch in range(num_epochs):
    gru_model.train()
    running_loss=0.0

    for idx,(input,labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs=gru_model(input)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if idx % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(dataloader):.4f}")


def classify_text_gru(text,model,char_to_index,max_len,index_to_label):
    indices=[char_to_index.get(char,0) for char in text[:max_len]]
    indices+=[0]*(max_len-len(indices))
    input_tensor=torch.tensor(indices,dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output=model(input_tensor)
        _,predicted_index=torch.max(output,1)#按照第一个维度求最大值，_表示第一个维度的最大值，predicted_index表示最大值对应的类别索引
        predicted_index=predicted_index.item()
        predicted_label=index_to_label[predicted_index]
        return predicted_label

index_to_label={i:label for label,i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")



