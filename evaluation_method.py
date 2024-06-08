import math
import numpy as np
import networkx as nx
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_sum_assignment
from collections import Counter
from scipy.special import comb
from itertools import combinations
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.cluster import KMeans
from networkx.algorithms import community


def NMI(com, real_com):
    """
    Compute the Normalized Mutual Information(NMI)
    Parameters
    --------
    com, real_com : list or numpy.array
        number of community of nodes
    """
    # if len(com) != len(real_com):
    #     return ValueError('len(A) should be equal to len(B)')

    com = np.array(com)
    real_com = np.array(real_com)
    total = len(com)
    com_ids = set(com)
    real_com_ids = set(real_com)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for id_com in com_ids:
        for id_real in real_com_ids:
            idAOccur = np.where(com == id_com)
            idBOccur = np.where(real_com == id_real)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py) + eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in com_ids:
        idAOccurCount = 1.0*len(np.where(com == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total + eps, 2)
    Hy = 0
    for idB in real_com_ids:
        idBOccurCount = 1.0*len(np.where(real_com == idB)[0])
        Hy = Hy - (idBOccurCount/total) * math.log(idBOccurCount/total + eps, 2)
    MIhat = 2.0*MI/(Hx + Hy)
    return MIhat

def modularity(G, community):
    """
    Compute modularity of communities of network
    Parameters
    --------
    G : networkx.Graph
        an undirected graph
    community : dict
        the communities result of community detection algorithms
    """
    V = [node for node in G.nodes()]
    m = G.size(weight='weight')  # if weighted
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    Q = 0
    for i in range(n):
        node_i = V[i]
        com_i = community[node_i]
        degree_i = G.degree(node_i)
        for j in range(n):
            node_j = V[j]
            com_j = community[node_j]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)
    return Q/(2 * m)


def f_same(cluA, cluB, clusters):
    S = np.matrix([[0 for i in range(clusters)] for j in range(clusters)])
    for i in range(len(cluA)):
        S[cluA[i], cluB[i]] += 1
    r = sum(S.max(0).T)
    c = sum(S.max(1))
    fsame = (r+c)/(float(len(cluA))*2)
    return fsame



def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def rand_score (labels_true, labels_pred):
    """given the true and predicted labels, it will return the Rand Index."""
    check_clusterings(labels_true, labels_pred)
    my_pair = list(combinations(range(len(labels_true)), 2)) #create list of all combinations with the length of labels.
    def is_equal(x):
        return (x[0]==x[1])
    my_a = 0
    my_b = 0
    for i in range(len(my_pair)):
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]]))
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == True):
                my_a += 1
            if(is_equal((labels_true[my_pair[i][0]],labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]]))
               and is_equal((labels_pred[my_pair[i][0]],labels_pred[my_pair[i][1]])) == False):
                my_b += 1
    my_denom = comb(len(labels_true),2)
    ri = (my_a + my_b) / my_denom
    return ri

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def jaccard(cluster, label):
    dist_cluster = np.abs(np.tile(cluster, (len(cluster), 1)) -
                          np.tile(cluster, (len(cluster), 1)).T)
    dist_label = np.abs(np.tile(label, (len(label), 1)) -
                        np.tile(label, (len(label), 1)).T)
    a_loc = np.argwhere(dist_cluster+dist_label == 0)
    n = len(cluster)
    a = (a_loc.shape[0]-n)/2
    same_cluster_index = np.argwhere(dist_cluster == 0)
    same_label_index = np.argwhere(dist_label == 0)
    bc = same_cluster_index.shape[0]+same_label_index.shape[0]-2*n-2*a
    return a/(a+bc)

def modularity_mod(G, community, A):
    """
    Compute modularity of communities of network
    Parameters
    --------
    G : networkx.Graph
        an undirected graph
    community : dict
        the communities result of community detection algorithms
    """
    V = [node for node in G.nodes()]
    m = G.size(weight='weight')  # if weighted
    n = G.number_of_nodes()
    #A = nx.to_numpy_array(G)
    print(A)
    Q = 0
    for i in range(n):
        node_i = V[i]
        com_i = community[node_i]
        degree_i = G.degree(node_i)
        for j in range(n):
            node_j = V[j]
            com_j = community[node_j]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)
    return Q/(2 * m)


def conductance(partition, G):
    """

    计算网络的传导率
    Parameters
    ----------
    partition 社区划分列表
    G 网络

    Returns
    -------
    average_conductance / count 网络平均传导率

    """
    # max_conductance = 0

    # for group in partition:
    #     if (len(group) == 1 or len(group) == len(graph.nodes())):
    #         continue

    #     group_conductance = nx.conductance(graph, group)
    #     if group_conductance > max_conductance:
    #         max_conductance = group_conductance

    # if (max_conductance == 0):
    #     return 1

    average_conductance = 0
    count = 0

    for group in partition:
        if (len(group) == 1 or len(group) == len(G.nodes())):
            average_conductance += 1
            count += 1
            continue

        average_conductance += nx.conductance(G, group)
        count += 1

    return average_conductance / count

def ndarray_to_partition(ndarray):
    """

    将聚类数组转化为社区划分
    Parameters
    ----------
    ndarray 聚类数组

    Returns
    -------
    partition 社区划分

    """
    ndarray = ndarray.tolist()
    # 设置重复元素列表
    duplicate_elements = []
    # 设置社区划分列表
    partition = []
    # 迭代
    for ndarray_iteration in range(len(ndarray)):
        # 元素第一次出现
        if ndarray[ndarray_iteration] not in duplicate_elements:
            # 将元素放入重复元素列表
            duplicate_elements.append(ndarray[ndarray_iteration])
            # 在社区划分列表中开辟一个新的空列表
            partition.append([])
            # 寻找重复元素列表中对应元素的下表，并将其存储至社区划分列表对应的列表中
            partition[duplicate_elements.index(ndarray[ndarray_iteration])].append(ndarray_iteration)
        # 重复元素出现
        else:
            partition[duplicate_elements.index(ndarray[ndarray_iteration])].append(ndarray_iteration)
    # 将社区划分列表中的迭代对象(list)转化为(set)
    partition = [set(n) for n in partition]
    return partition

if __name__ == "__main__":
    real_result = pd.read_csv('./data/karate_realresult.csv', header=None)
    real_com = real_result.values.T[0]
    edge_list = pd.read_csv('./data/karate_edges.csv', header=None)
    edges = list(zip(*edge_list.values.T))
    node_num = 34
    G = nx.Graph()
    G.add_nodes_from(list(range(0, node_num - 1)))
    G.add_edges_from(edges)
    adj = np.array(nx.adjacency_matrix(G).todense())
    km = KMeans(n_clusters=2)
    km.fit(adj)
    kmlb = km.labels_
    partition = ndarray_to_partition(kmlb)
    (coverage, performance) = community.partition_quality(G, partition)
    conductance = conductance(partition, G)






