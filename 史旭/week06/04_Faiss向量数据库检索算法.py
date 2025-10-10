import time

import numpy
import numpy as np
from faiss import IndexFlatL2, IndexIVFPQ

# Faiss向量数据库的检索算法（暴力检索 IndexFlatL2  和  PQ量化检索 IndexIVFPQ）


# 模拟向量数据库，准备数据
print("-----  模拟向量数据库，准备数据  -----")

# 通过 numpy 创建向量数据
data_size = 10000  # 向量数据库 总数据量
query_size = 10  # 检索向量数量
dim = 128  # 向量维度

# 设置随机种子，保证每次运行生成的 向量一致
np.random.seed(1234)
data = np.random.random((data_size, dim))
query = np.random.random((query_size, dim))

print("-----  向量数据信息生成完成  -----")

# 1.暴力检索（将要检索的向量 与 向量数据库中的所有向量，逐个计算相似度，获取topN（属于 精确检索，效率低）
# ①创建 暴力检索对象
indexFlatL2 = IndexFlatL2(dim)

# ②将 向量数据库信息 添加到 IndexFLatL2
indexFlatL2.add(data)

# ③暴力检索（循环便利 计算相似度）
# 返回两个结果，每个查询向量与所有向量之间的举例 以及 对应的索引位置
start_time = time.time()
d_indexFlatL2, i_indexFlatL2 = indexFlatL2.search(query, 4)
end_time = time.time()

print(f"IndexFlatL2数据量：{indexFlatL2.ntotal}")
print("IndexFlatL2检索结果--距离关系：")
print(d_indexFlatL2)
print("IndexFlatL2检索结果--对应索引位置：")
print(i_indexFlatL2)
print(f"IndexFlatL2检索耗时：{end_time - start_time}ms")

print("-----  暴力检索结束  -----")
print("-" * 50)

# 2.PQ检索 IndexIVFPQ（近似搜索，精度稍差，效率更高）
# 先根据 IndexFlatL 量化器，通过K-Mearns对向量数据进行聚类操作，划分簇（倒排索引记录）
# 之后划分子向量N块，每一块包含所有向量对应位置的字向量（通过训练 构建一个码本 例如256维，即256个分类）
# 然后对每个字向量根据码本进行分类，此时每个字向量不再是一个n维的向量，而是一个分类结果（一个字节，即对应分类的索引位置）
# 因此，存储的大小缩小了数十倍，甚至更高
# 在检索时，先判断检索向量在那几簇中，再对查询向量划分子向量，然后计算每个子向量与对应码本分类的相似性分数，得到 n*256 的二维向量
# 最后，让簇中的向量，参照 n*256 二维向量，计算n个子向量分数综合，最后将结果排序，得出分数最高的三个向量，并返回原始索引位置

# ①创建 量化器
flat = IndexFlatL2(dim)

# ②创建 IndexIVFPQ 对象
# 量化器  原始维度  聚类的簇大小  子向量大小  码本大小（几个字节）
indexIVFPQ = IndexIVFPQ(flat, dim, 100, 8, 8)

# ③通过 向量数据库 进行训练（聚类划分簇，构建码本）
indexIVFPQ.train(data)

# ④将向量数据库信息 添加到 IndexIVFPQ
indexIVFPQ.add(data)

# ⑤检索
# 检索向量  topN
start_time = time.time()
d_indexIVFPQ, i_indexIVFPQ = indexIVFPQ.search(query, 4)
end_time = time.time()

print(f"IndexIVFPQ数据量：{indexIVFPQ.ntotal}")
print("IndexIVFPQ检索结果--距离关系：")
print(d_indexIVFPQ)
print("IndexIVFPQ检索结果--对应索引位置：")
print(i_indexIVFPQ)
print(f"IndexIVFPQ检索耗时：{end_time - start_time}ms")

print("-----  PQ检索结束  -----")
print("-" * 50)

print(f"IndexFlatL2检索--第一条结果：{i_indexFlatL2[0]}")
print(f"IndexIVFPQ检索--第一条结果：{i_indexIVFPQ[0]}")
