import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import Word2Vec  # NLP 语言模型
import random

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)


# 1. 创建一个有逻辑意义的社交网络图
def create_social_network():
    # 模拟创建的图， networkx

    G = nx.Graph()

    # 创建三个社交圈子：研究人员、艺术家和运动员
    researchers = [f"Researcher_{i}" for i in range(15)]
    artists = [f"Artist_{i}" for i in range(15)]
    athletes = [f"Athlete_{i}" for i in range(15)]

    # 添加节点
    for node in researchers + artists + athletes:
        G.add_node(node)

    # 在每个圈子内创建紧密连接
    # 研究人员圈子
    for i, researcher in enumerate(researchers):
        # 连接同一领域的研究人员
        for j in range(i + 1, len(researchers)):
            if np.random.random() < 0.4:  # 40%的概率连接
                G.add_edge(researcher, researchers[j])

    # 艺术家圈子
    for i, artist in enumerate(artists):
        for j in range(i + 1, len(artists)):
            if np.random.random() < 0.5:  # 50%的概率连接
                G.add_edge(artist, artists[j])

    # 运动员圈子
    for i, athlete in enumerate(athletes):
        for j in range(i + 1, len(athletes)):
            if np.random.random() < 0.6:  # 60%的概率连接
                G.add_edge(athlete, athletes[j])

    # 添加一些跨圈子的连接（较少）
    for researcher in researchers[:5]:  # 前5个研究人员
        for artist in artists[:3]:  # 前3个艺术家
            if np.random.random() < 0.3:  # 30%的概率连接
                G.add_edge(researcher, artist)

    for artist in artists[5:10]:  # 中间5个艺术家
        for athlete in athletes[5:10]:  # 中间5个运动员
            if np.random.random() < 0.3:  # 30%的概率连接
                G.add_edge(artist, athlete)

    for researcher in researchers[10:]:  # 后5个研究人员
        for athlete in athletes[10:]:  # 后5个运动员
            if np.random.random() < 0.2:  # 20%的概率连接
                G.add_edge(researcher, athlete)

    return G, researchers, artists, athletes


# 2. 生成随机游走序列（DeepWalk的核心）
def generate_random_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append([str(x) for x in walk])
    return walks


# 3. 使用Word2Vec学习节点表示
def learn_embeddings(walks, dimensions=128, window_size=5):
    # 输入文本序列  -》 词向量
    # 输入图节点序列  -》 节点向量
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,  # skip-gram
        workers=4,
        epochs=10
    )
    return model


# 4. 计算节点相似度矩阵
def calculate_similarity_matrix(model, nodes):
    embeddings = np.array([model.wv[node] for node in nodes])
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


# 5. 可视化函数
def visualize_graph(G, researchers, artists, athletes, similarity_matrix, top_n=5):
    # 设置节点颜色
    node_colors = []
    for node in G.nodes():
        if node in researchers:
            node_colors.append('red')
        elif node in artists:
            node_colors.append('blue')
        else:
            node_colors.append('green')

    plt.figure(figsize=(15, 5))

    # 子图1: 原始网络结构
    plt.subplot(131)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=node_colors, node_size=50, with_labels=False)
    plt.title("Original Network Structure")

    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Researchers'),
        Patch(facecolor='blue', label='Artists'),
        Patch(facecolor='green', label='Athletes')
    ]
    plt.legend(handles=legend_elements, loc='best')

    # 子图2: 节点嵌入的2D投影
    plt.subplot(132)
    # 获取所有节点的嵌入向量
    all_nodes = list(G.nodes())
    embeddings = np.array([model.wv[node] for node in all_nodes])

    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # 绘制2D投影
    for i, node in enumerate(all_nodes):
        color = 'red' if node in researchers else 'blue' if node in artists else 'green'
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=color, s=50)
        plt.annotate(node, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title("Node Embeddings (PCA)")

    # 子图3: 节点相似度热力图
    plt.subplot(133)
    # 选择部分节点显示热力图
    selected_nodes = researchers[:3] + artists[:3] + athletes[:3]
    selected_indices = [all_nodes.index(node) for node in selected_nodes]
    selected_similarity = similarity_matrix[np.ix_(selected_indices, selected_indices)]

    plt.imshow(selected_similarity, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(selected_nodes)), selected_nodes, rotation=45)
    plt.yticks(range(len(selected_nodes)), selected_nodes)
    plt.title("Node Similarity Heatmap")

    plt.tight_layout()
    plt.show()

    # 打印最相似的节点对
    print("\nTop 5 most similar node pairs:")
    all_nodes = list(G.nodes())
    similarities = []
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            similarities.append((all_nodes[i], all_nodes[j], similarity_matrix[i, j]))

    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)

    for i in range(min(top_n, len(similarities))):
        node1, node2, sim = similarities[i]
        print(f"{node1} - {node2}: {sim:.4f}")


# 主程序
if __name__ == "__main__":
    # 创建图
    G, researchers, artists, athletes = create_social_network()
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # 生成随机游走
    walks = generate_random_walks(G, num_walks=10, walk_length=20)
    print(f"Generated {len(walks)} random walks")

    # 学习节点嵌入
    model = learn_embeddings(walks, dimensions=64, window_size=5)
    print("Node embeddings learned")

    # 计算相似度矩阵
    all_nodes = list(G.nodes())
    similarity_matrix = calculate_similarity_matrix(model, all_nodes)

    # 可视化结果
    visualize_graph(G, researchers, artists, athletes, similarity_matrix)
