import pandas as pd
import networkx as nx
import auto_encode as ae
import numpy as np
import evaluation_method as em
import random
from networkx.algorithms import community
import time
from collections import defaultdict, namedtuple
from sklearn.mixture import GaussianMixture as GMM
# 矩阵转置翻转
def matrix_transpose_flip(adj):
    adj_T = adj.T
    adj_res = adj + adj_T
    return adj_res

def random_delete_edges(G, missing_ratio):
    """
    模拟网络边缺失过程
    Parameters
    ----------
    G 原始网络
    missing_ratio 缺失比例因子

    Returns
    -------
    G 删除随机边后的网络

    """
    missing_edges = []
    max_edges = list(G.edges)
    edges_length = len(G.edges)

    for i in range(int(edges_length * missing_ratio)):
        missing_edge = random.choice(max_edges)
        missing_edges.append(missing_edge)
    G.remove_edges_from(missing_edges)

    return G

def uwg_normalization(G, sim):
    # 权重归一化
    adj = np.array(nx.adjacency_matrix(G).todense())
    ele_max = adj.max()
    adj_norm = adj * (1 / ele_max)
    sim_norm = np.array(sim)
    norm_res = np.multiply(adj_norm, sim_norm)
    return norm_res

def k_core_s(G):
    """

    k-core 算法
    Parameters
    ----------
    G 网络图

    Returns
    -------
    sim 相似度矩阵

    """
    """
    测试用例 strike数据集
    edge_list = pd.read_csv('../data/strike_edges.csv', header=None)
    edges = list(zip(*edge_list.values.T))

    G = nx.Graph()
    G.add_nodes_from(list(range(0, 23)))
    G.add_edges_from(edges)
    """
    # 生成邻接矩阵
    adj = np.array(nx.adjacency_matrix(G).todense())
    # 获取节点core数目列表
    core_num = nx.core_number(G)
    # print(core_num)
    # 获取最大节点core数目列表
    max_core_num = max(list(core_num.values()))
    adj = adj.tolist()
    # 归一化
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i > j:
                adj[i][j] = 0
            elif adj[i][j] != 0:
                adj[i][j] = (adj[i][j] * core_num[i]) / max_core_num
    adj = np.array(adj)
    # 矩阵转置翻转
    sim = matrix_transpose_flip(adj)
    return sim

def collapse(G, grouped_nodes):
    # 图压缩
    mapping = {}
    members = {}
    C = G.__class__()
    i = 0  # required if G is empty
    remaining = set(G.nodes())
    for i, group in enumerate(grouped_nodes):
        group = set(group)
        assert remaining.issuperset(
            group
        ), "grouped nodes must exist in G and be disjoint"
        remaining.difference_update(group)
        members[i] = group
        mapping.update((n, i) for n in group)
    # remaining nodes are in their own group
    for i, node in enumerate(remaining, start=i + 1):
        group = {node}
        members[i] = group
        mapping.update((n, i) for n in group)
    number_of_groups = i + 1
    C.add_nodes_from(range(number_of_groups))
    C.add_edges_from(
        (mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v]
    )
    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, name="members", values=members)
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    return C

def unconstrained_one_edge_augmentation(G):
    ccs1 = list(nx.connected_components(G))
    C = collapse(G, ccs1)
    # 创建生成树
    meta_nodes = list(C.nodes())
    # 创建路径
    meta_aug = list(zip(meta_nodes, meta_nodes[1:]))
    # 将路径映射到原始图形
    inverse = defaultdict(list)
    for k, v in C.graph["mapping"].items():
        inverse[v].append(k)
    for mu, mv in meta_aug:
        yield (inverse[mu][0], inverse[mv][0])

def k_edge_augmentation(G, k):
    if k <= 0:
        raise ValueError(f"k must be a positive integer, not {k}")
    elif k == 1:
        return unconstrained_one_edge_augmentation(G)

def data_to_save(res):
    #  保存实验结果
    data = pd.DataFrame(res)
    data.to_csv(SAVE_PATH, mode="a+", header=None, index=None)

def GMM_mixture(x, clusters):
    gmm = GMM(
        n_components=clusters,
        covariance_type="full",
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=300,
        n_init=10,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ).fit(x)
    label = gmm.predict(x)
    return label

def missing_edges_community_detection(DATESET_STR, MISSING_FACTOR, AUGMENTATION_FACTOR, DATESET_INFORMATION):
    """

    缺失边缘社区检测算法
    Parameters
    ----------
    DATESET_STR 数据集名称
    MISSING_FACTOR 缺失因子
    AUGMENTATIION_FACTOR 增强因子

    Returns
    -------
    res 社区检测算法各评价指标

    """
    try:
        flag = 0
        real_result = pd.read_csv('../data/' + str(DATESET_STR) + '_realresult.csv', header=None)
        edge_list = pd.read_csv('../data/' + str(DATESET_STR) + '_edges.csv', header=None)
        real_com = real_result.values.T[0]

    except:
        flag = 1
        edge_list = pd.read_csv('../data2/' + str(DATESET_STR) + '_edges.csv', header=None)

    edges = list(zip(*edge_list.values.T))
    clusters = DATESET_INFORMATION[DATESET_STR]["clusters"]
    node_num = DATESET_INFORMATION[DATESET_STR]["node_num"]

    start = time.perf_counter()

    G = nx.Graph()
    G.add_nodes_from(list(range(0, node_num - 1)))

    # WEIGHTED
    # G.add_weighted_edges_from(edges)
    # weights = G.edges.data("weight")
    # for k in weights:
    #     G.add_edge(int(k[0]), int(k[1]), weight=k[2])

    G.add_edges_from(edges)

    # density = nx.density(G)
    # print(density)
    # degrees = [val for (node, val) in G.degree()]
    # d = sum(degrees) / len(degrees)

    G = random_delete_edges(G, MISSING_FACTOR)

    edges_augmentation = list(sorted(k_edge_augmentation(G, k=AUGMENTATION_FACTOR)))
    #edges_augmentation = list(sorted(nx.k_edge_augmentation(G, k=AUGMENTATION_FACTOR)))
    G.add_edges_from(edges_augmentation)

    # adj = np.array(nx.adjacency_matrix(G).todense())
    # sim = adj
    G.remove_edges_from(nx.selfloop_edges(G))
    sim = k_core_s(G)

    # WEIGHTED
    # sim = uwg_normalization(G, sim)

    trained_start = time.perf_counter()
    x = ae.variational_auto_encoder(sim, DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][0],
                                    DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][1])
    trained_end = time.perf_counter()

    kmlb = GMM_mixture(x, clusters)

    end = time.perf_counter()
    sum_time = end - start
    non_trained_time = (trained_start - start) + (end - trained_end)

    if flag == 1:
        Q = em.modularity(G, kmlb)
        print('Q=', float(Q))
        partition = em.ndarray_to_partition(kmlb)
        (coverage, performance) = community.partition_quality(G, partition)
        print('coverage=', float(coverage))
        print('performance=', float(performance))
        conductance = em.conductance(partition, G)
        print('conductance=', float(conductance))
        print('sum_time=', float(sum_time))
        print('non_trained_time=', float(non_trained_time))
        print('edges_augmentation=', len(edges_augmentation))
        if node_num <= 1000:
            print(G.edges)
        res = np.array(
            [Q, coverage, performance, conductance, sum_time, non_trained_time, clusters,
             str([DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][0],
                  DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][1]]), "none + ae"]
            ).reshape(1, 9)

        return res

    else:
        nmi = em.NMI(kmlb, real_com)
        print('NMI=', float(nmi))
        acc = em.acc(real_com, kmlb)
        print('ACC=', float(acc))
        rand_score = em.rand_score(real_com, kmlb)
        print('rand_score=', float(rand_score))
        purity_score = em.purity_score(real_com, kmlb)
        print('purity_score=', float(purity_score))
        fsame = em.f_same(kmlb, real_com, clusters)
        print('Fsame=', float(fsame))
        Q = em.modularity(G, kmlb)
        print('Q=', float(Q))
        partition = em.ndarray_to_partition(kmlb)
        (coverage, performance) = community.partition_quality(G, partition)
        print('coverage=', float(coverage))
        print('performance=', float(performance))
        conductance = em.conductance(partition, G)
        print('conductance=', float(conductance))
        print('sum_time=', float(sum_time))
        print('non_trained_time=', float(non_trained_time))
        print('edges_augmentation=', len(edges_augmentation))
        if node_num <= 1000:
            print(G.edges)
        res = np.array([nmi, acc, rand_score, purity_score, float(fsame), Q, coverage, performance, conductance, sum_time, non_trained_time,
                        str([DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][0],
                             DATESET_INFORMATION[DATESET_STR]["neuron_network_parameters"][1]]), str(AUGMENTATION_FACTOR)]
                       ).reshape(1, 13)

        return res

if __name__ == "__main__":

    DATESET_STR = 'karate'
    MISSING_FACTOR = 0.4
    AUGMENTATION_FACTOR = 1
    LOOP_COUNT = 1
    SAVE_PATH = './res/' + str(DATESET_STR) + '_' + str(MISSING_FACTOR) + '_result.csv'

    DATESET_INFORMATION = {
        "karate": {"node_num": 34, "clusters": 2, "neuron_network_parameters": [[24], 18]},
        "dolphins": {"node_num": 62, "clusters": 2, "neuron_network_parameters": [[32, 24], 12]},
    }

    for i in range(LOOP_COUNT):
        res = missing_edges_community_detection(DATESET_STR, MISSING_FACTOR, AUGMENTATION_FACTOR, DATESET_INFORMATION)
        data_to_save(res)














