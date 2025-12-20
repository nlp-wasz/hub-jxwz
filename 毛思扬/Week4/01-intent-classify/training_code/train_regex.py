import jieba
from collections import Counter
import re
import pandas as pd

# 读取数据集
dataset = pd.read_csv("assets/dataset/作业数据-waimai_10k.csv")
print(dataset.head())



# 定义停用词列表
stop_words = pd.read_csv('assets/dataset/baidu_stopwords.txt', header=None)[0].values
# 数据预处理函数
def preprocess_text(text):
    # 转换为字符串并去除标点符号和数字，只保留中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    return cleaned_text

# 分词并过滤停用词的函数
def tokenize_and_filter(text):
    # 中文分词
    seg_list = jieba.lcut(text, cut_all=False)
    # 过滤停用词和单字词
    filtered_words = [word for word in seg_list if word not in stop_words and len(word) > 1]
    return filtered_words

# 处理好评数据 (label=1)
positive_reviews = dataset[dataset['label'] == 1]['review']
positive_texts = ' '.join(positive_reviews.astype(str))
cleaned_positive_text = preprocess_text(positive_texts)
positive_words = tokenize_and_filter(cleaned_positive_text)
positive_word_counts = Counter(positive_words)
top_50_positive = positive_word_counts.most_common(50)

print("好评(label=1)中出现频率最高的前50个词：")
positive_words_list = []
for word, count in top_50_positive:
    print(f"{word}: {count}")
    positive_words_list.append(word)

print("\n" + "="*50 + "\n")

# 处理差评数据 (label=0)
negative_reviews = dataset[dataset['label'] == 0]['review']
negative_texts = ' '.join(negative_reviews.astype(str))
cleaned_negative_text = preprocess_text(negative_texts)
negative_words = tokenize_and_filter(cleaned_negative_text)
negative_word_counts = Counter(negative_words)
top_50_negative = negative_word_counts.most_common(50)

print("差评(label=0)中出现频率最高的前50个词：")
negative_words_list = []
for word, count in top_50_negative:
    negative_words_list.append(word)
    print(f"{word}: {count}")


# 去除两个列表中的重复词
common_words = set(positive_words_list) & set(negative_words_list)

# 从两个列表中移除共同词
unique_positive_words = [word for word in positive_words_list if word not in common_words]
unique_negative_words = [word for word in negative_words_list if word not in common_words]

print("去重后的好评关键词列表：", unique_positive_words)
print("去重后的差评关键词列表：", unique_negative_words)