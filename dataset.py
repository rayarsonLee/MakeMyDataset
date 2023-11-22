import os.path

import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from node2vec import Node2Vec
import pandas as pd


def generate_probability_sequence_with_type(length, inflection_point, event_type):
    """
    创建长度为length的数值序列，该序列先上升，再下降，拐点位于0~length,数值范围在0~1之间，用于表示概率值
    :param length: 待生成序列的长度
    :param inflection_point: 生成的先上升，再下降的序列的拐点
    :param event_type: 时间序列所属类别
    :return: 返回长度为length，拐点为turning_point的数值序列
    """
    lower_bound = 0
    upper_bound = 1

    # 创建序列前半段递增的序列
    start = 0
    end = 0.7
    # 生成正态分布的随机数
    # 正太分布均值
    mean_value = (start + end) / 2
    # 标准差
    std_dev = (end - start) / 4
    increasing_sequence = np.random.normal(mean_value, std_dev, inflection_point)
    increasing_sequence = np.sort(increasing_sequence)
    increasing_sequence = np.clip(increasing_sequence, lower_bound, upper_bound)
    # 创建序列后半段递减的序列
    start = increasing_sequence[-1]
    end = 1
    # 生成正态分布的随机数
    # 正太分布均值
    mean_value = (start + end) / 2
    # 标准差
    std_dev = (end - start) / 4
    decreasing_sequence = np.random.normal(mean_value, std_dev, length - inflection_point)
    decreasing_sequence = np.sort(decreasing_sequence)[::-1]
    decreasing_sequence = np.clip(decreasing_sequence, lower_bound, upper_bound)
    # 添加均匀分布的噪声
    # 噪声的强度delta
    delta = 0.05
    uniform_noise = np.random.uniform(-delta, delta, length)
    # 将递增序列和递减序列合并
    result_sequence = np.concatenate((increasing_sequence, decreasing_sequence))
    result_sequence += uniform_noise
    # clamp数组中的值

    clamp_result_sequence = np.clip(result_sequence, lower_bound, upper_bound)
    result = (clamp_result_sequence, event_type)
    return result


def generate_dataset_sequence(number_of_dataset, length, type_num):
    """
    生成时间序列数据集
    :param number_of_dataset: 数据集中时间序列的数量
    :param length: 每个时间序列的长度
    :param type_num: 不同不确定事件种类的总数量
    :return: 生成的时间序列数据集
    """
    result = []
    for cur_type in range(1, type_num + 1):
        for _ in range(0, number_of_dataset):
            inflection_point = cur_type
            cur = generate_probability_sequence_with_type(length, inflection_point, cur_type)
            result.append(cur)
    return result


def save_dataset_sequence_file(dataset_sequence):
    """
    将带标签的数据集序列保存到CSV文件中
    :param dataset_sequence: 带标签的数据集序列
    :return:
    """
    csv_file_path = "dataset_sequence.csv"
    with open(csv_file_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 创建标题行
        table_name = []
        temporal_series_length = len(dataset_sequence[0][0])
        for index in range(temporal_series_length):
            table_name.append("time" + str(index))
        table_name.append("type")
        csv_writer.writerow(table_name)

        for item in dataset_sequence:
            it = list(item[0]) + [item[1]]
            csv_writer.writerow(it)


def check_edge_exists_between(i, j, data):
    """
    在生成可见性图时，判断时间序列的两个节点i和j之间是否应该存在边
    :param i: 节点i
    :param j: 节点j
    :param data: 时间序列
    :return: i,j之间是否存在边
    """
    k = (data[j] - data[i]) / (j - i)
    for cur in range(i + 1, j, 1):
        hi = data[i] + k * (cur - i)
        if hi < data[cur]:
            return False
    return True


def show_graph_by_plt(graph):
    """
    输出graph
    :param graph:
    :return:
    """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color="green", font_size=10, font_color='black')
    node_labels = nx.get_node_attributes(graph, 'weight')
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def convert_sequence_to_graph(temporal_sequence):
    """
    将时间序列转化为图的结构
    :param temporal_sequence:
    :return:
    """
    graph = nx.Graph()
    for index in range(len(temporal_sequence)):
        graph.add_node(index, weight=temporal_sequence[index])
    edges = []
    for i in range(len(temporal_sequence) - 1):
        for j in range(i + 1, len(temporal_sequence)):
            if check_edge_exists_between(i, j, temporal_sequence):
                edges.append((i + 1, j + 1))
    graph.add_edges_from(edges)
    # 将图转化为邻接矩阵
    return graph


def graph_embedding(G):
    """
    计算图G的embedding
    :param G:
    :return:
    """
    # 获取节点权值
    node_weights = dict(G.nodes(data='weight'))

    # 使用node2vec获取节点嵌入
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 获取节点嵌入
    node_embeddings = {node: model.wv[node] for node in G.nodes}

    # 将节点嵌入和节点权重拼接
    vectorized_nodes = np.array([np.concatenate([node_embeddings[node], [node_weights[node]]]) for node in G.nodes])

    # vectorized_nodes 现在包含了每个节点的向量表示，其中包括节点嵌入和节点权重
    return vectorized_nodes


def save_embedding(embedding, index):
    """
    将graph的一个embedding保存
    :param embedding:
    :param index:
    :return:
    """
    folder_path = "embeddings"
    filepath = os.path.join(folder_path, f"node_{index}_embedding.npy")
    np.save(filepath, embedding)


def save_embeddings(features):
    """
    保存每一个feature的embedding
    :param features:
    :return:
    """
    for index in range(len(features)):
        print(index)
        graph = convert_sequence_to_graph(features[index])
        embedding = graph_embedding(graph)
        save_embedding(embedding, index)


def load_dataset_from_embedding_files():
    """
    从embeddings路径中读取图嵌入对应的1000个npy文件，并将他们保存为tensor
    :return: 返回一个[1000，21，65]维度的tensor。
    """
    folder_path = './embeddings/'
    file_template = 'node_{}_embedding.npy'
    tensor_list = []
    for i in range(1000):
        file_path = folder_path + file_template.format(i)
        array_data = np.load(file_path, allow_pickle=True)
        array_data = array_data.astype(np.float32)
        tensor_data = torch.tensor(array_data, dtype=torch.float32)
        tensor_list.append(tensor_data)
    data_tensor = torch.stack(tensor_list)
    return data_tensor


def make_train_loader():
    csv_file_path = './dataset_sequence.csv'
    labels_df = pd.read_csv(csv_file_path)
    features_list = []
    labels_list = []
    for i in range(1000):
        npy_folder_path = './embeddings/'
        file_template = 'node_{}_embedding.npy'
        npy_file_path = npy_folder_path + file_template.format(i)
        features_data = np.load(npy_file_path, allow_pickle=True)
        label = labels_df.iloc[i, -1]
        features_list.append(features_data)
        labels_list.append(label)
    features_list = np.array(features_list)
    labels_list = np.array(labels_list)
    features_list = features_list.astype(np.float32)
    labels_list = labels_list.astype(np.int32)
    features_tensor = torch.stack([torch.tensor(data, dtype=torch.float32) for data in features_list])
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    dataset = TensorDataset(features_tensor, labels_tensor)
    batch_size = 32  # 根据需要调整批次大小
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"{batch_idx}/{inputs.shape}:{labels.shape}")


# 创建数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
