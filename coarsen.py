"""
Matching methods for graph coarsening.
"""
import numpy as np
from graph import Graph
from utils import cmap2C
# from similarity import *
from collections import defaultdict
from scipy.stats import entropy


def normalized_adj_wgt(ctrl, graph):
    # sens_num = graph.sens_num
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    # attr_dist = graph.attr_dist
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree  # sum of incident edges' weights of each node
    # node_wgt = graph.node_wgt  # number of nodes in the supernode
    for i in range(graph.node_num):
        # attr_i = attr_dist[i]
        # wgt_idx = node_wgt[i]
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt


def mile_match(ctrl, graph):
    """
    Matching method in MILE, with no fairness considered
    :param ctrl:
    :param graph:
    :return:
    """
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(ctrl, graph)

    max_node_wgt = ctrl.max_node_wgt

    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(ctrl, graph, matched, groups)
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if (not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt:
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node.
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return groups, coarse_graph_size


def jaccard_idx_preprocess(ctrl, graph, matched, groups):
    """
    Structure-Equivalent Matching in MILE (Liang et al, 2021)
    Use hashmap to find out nodes with exactly same neighbors.
    :param ctrl:
    :param graph:
    :param matched:
    :param groups:
    :return:
    """
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        neighs2node[neighs].append(i)
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def general_match(ctrl, graph):
    # Basics
    node_num = graph.node_num
    edge_num = graph.edge_num
    sens_num = graph.sens_num
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
    attr_dist = graph.attr_dist
    norm_attr_dist = graph.norm_attr_dist
    attr_range = graph.attr_range
    attr_dim = attr_range[-1]
    degree = graph.degree
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(ctrl, graph)
    max_node_wgt = ctrl.max_node_wgt
    groups = []
    matched = [False] * node_num

    # match
    num_neighbors = [adj_idx[i + 1] - adj_idx[i] for i in range(node_num)]
    sorted_idx = np.argsort(num_neighbors)
    wgt_merge = ctrl.wgt_merge
    # count_mix_attr = 0
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        attr_i = attr_dist[idx]
        norm_attr_i = norm_attr_dist[idx]
        wgt_idx = node_wgt[idx]
        # weight_fair = wgt_idx - attr_i
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx or matched[neigh]:
                continue
            curr_wgt = (1 - wgt_merge) * norm_adj_wgt[j] + \
                       wgt_merge * ctrl.coarse_fair_func(ctrl, attr_i, attr_dist[neigh],
                                                   norm_attr_i, norm_attr_dist[neigh],
                                                   wgt_idx, node_wgt[neigh], sens_num)
            if max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt:
                max_idx = neigh
                max_wgt = curr_wgt
        matched[idx] = matched[max_idx] = True
        # count_mix_attr += 1 if np.all(np.equal(attr_i, attr_dist[max_idx])) else 0
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])

    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    # print(f'Mix intra-group nodes / all nodes {count_mix_attr}/{coarse_graph_size}')
    return groups, coarse_graph_size


def create_coarse_graph(ctrl, graph, groups, coarse_graph_size):
    '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
    coarse_graph = Graph(coarse_graph_size, graph.edge_num, graph.sens_num, graph.sens_dim, graph.attr_range)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
    attr_dist = graph.attr_dist
    norm_attr_dist = graph.norm_attr_dist

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree
    coarse_attr_dist = coarse_graph.attr_dist
    coarse_norm_attr_dist = coarse_graph.norm_attr_dist

    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    for idx in range(len(groups)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]
            coarse_attr_dist[coarse_node_idx] += attr_dist[merged_node]  # * node_wgt[merged_node]
            coarse_norm_attr_dist[coarse_node_idx] = coarse_attr_dist[coarse_node_idx] / coarse_node_wgt[coarse_node_idx]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop.
                coarse_degree[coarse_node_idx] += adj_wgt[j]
        # coarse_attr_dist[coarse_node_idx] /= coarse_node_wgt[coarse_node_idx]
        coarse_adj_idx[coarse_node_idx + 1] = nedges

    coarse_graph.edge_num = nedges
    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    return coarse_graph


"""
Define your attribute divergence functions here.
You need to manually add them to argument parser in `main.py`. 
"""


def kl_dvg(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    """
    Compute the KL-divergence of two normalized attribute vector.
    Note that the attribute vectors must be normalized (sum to 1) before calling it.
    """
    return 1 - 1 / (1 + entropy(norm1, norm2))


def mergefair_group(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    return attr2.dot(wgt1 - attr1) / sens_num / (wgt1 * wgt2)


def abs_diff(ctrl, attr1, attr2, norm1, norm2, wgt1, wgt2, sens_num):
    return np.sum(np.abs(attr1 - attr2)) / sens_num / (wgt1 + wgt2)
