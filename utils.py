"""
utils
"""
import logging
import sys
from graph import Graph
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import time
'''
Packages required for link prediction datasets
'''
import os
import networkx as nx
import pickle as pkl
from typing import Dict, Tuple

def setup_custom_logger(name):
    """Set up the logger, from MILE """
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)

    return logger


class Mapping:
    """Used for mapping index of nodes since the data structure used for graph requires continuous index."""
    def __init__(self, old2new, new2old):
        self.old2new = old2new
        self.new2old = new2old


"""
read_data: Read the dataset from the disk.
"""


def build_graph_after_load(edges, sens):
    # Basic statistics
    node_num = len(edges.indptr) - 1
    edge_num = len(edges.indices)
    sens_num = sens.shape[1]
    sens_dim = 0
    attr_range = np.zeros(sens_num + 1, dtype=np.int32)
    for j in range(sens_num):
        attr_range[j] = sens_dim
        sens_dim += len(np.unique(sens[:, j]))
    attr_range[sens_num] = sens_dim
    graph = Graph(node_num, edge_num, sens_num, sens_dim, attr_range)

    # Update graph
    graph.adj_list = edges.indices
    graph.adj_idx = edges.indptr
    graph.adj_wgt = edges.data
    for i in range(node_num):
        graph.node_wgt[i] = 1
        graph.degree[i] = np.sum(graph.get_neigh_edge_wgts(i))
        for j in range(sens_num):
            graph.norm_attr_dist[i, attr_range[j] + sens[i, j]] = graph.attr_dist[i, attr_range[j] + sens[i, j]] = 1
    return graph


def read_data(args):
    graph = None
    if args.task == 'nc':
        if args.format == 'nifty':
            # Load files
            edges = sp.load_npz(f'datasets/{args.data}/edges.npz')
            features = np.load(f'datasets/{args.data}/features.npy')
            labels = np.load(f'datasets/{args.data}/labels.npy')
            sens = np.load(f'datasets/{args.data}/sens.npy')
            graph = build_graph_after_load(edges, sens)
            return graph, features, labels, sens
        else:
            raise NotImplementedError
    else:
        path = os.path.join('datasets', args.data, str(args.seed))
        if os.path.isdir(path):
            '''
            Read pre-processed data.
            '''
            edges = sp.load_npz(os.path.join(path, 'edges.npz'))
            features = np.load(os.path.join(path, 'features.npy'))
            sens = np.load(os.path.join(path, 'sens.npy'))
            train_edges_true = np.load(os.path.join(path, 'train_edges_true.npy'))
            train_edges_false = np.load(os.path.join(path, 'train_edges_false.npy'))
            test_edges_true = np.load(os.path.join(path, 'test_edges_true.npy'))
            test_edges_false = np.load(os.path.join(path, 'test_edges_false.npy'))
            train_edges_true, train_edges_false = [tuple(x) for x in train_edges_true], [tuple(x) for x in
                                                                                      train_edges_false]
            test_edges_true, test_edges_false = [tuple(x) for x in test_edges_true], [tuple(x) for x in test_edges_false]
        else:
            os.mkdir(path)
            if args.data == 'cora':
                G, edges, features, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, _ = cora(seed=args.seed)
                sens = sensitive.reshape(-1, 1)
            elif args.data == 'citeseer':
                G, edges, features, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, _ = citeseer(seed=args.seed)
                sens = sensitive.reshape(-1, 1)
            elif args.data == 'pubmed':
                G, edges, features, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, _ = pubmed(seed=args.seed)
                sens = sensitive.reshape(-1, 1)
            elif args.data == 'pokec-n':
                G, edges, features, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, _ = pokec(seed=args.seed)
            else:
                raise NotImplementedError
            np.save(os.path.join(path, 'features.npy'), features)
            np.save(os.path.join(path, 'sens.npy'), sens)
            sp.save_npz(os.path.join(path, 'edges.npz'), edges)
            np.save(os.path.join(path, 'train_edges_true.npy'), np.array(train_edges_true))
            np.save(os.path.join(path, 'train_edges_false.npy'), np.array(train_edges_false))
            np.save(os.path.join(path, 'test_edges_true.npy'), np.array(test_edges_true))
            np.save(os.path.join(path, 'test_edges_false.npy'), np.array(test_edges_false))
        graph = build_graph_after_load(edges, sens)

        return graph, features, sens, train_edges_true, train_edges_false, test_edges_true, test_edges_false


def normalized(embeddings, per_feature=True):
    if per_feature:
        scaler = MinMaxScaler()
        scaler.fit(embeddings)
        return scaler.transform(embeddings)
    else:
        return normalize(embeddings, norm='l2')


def graph_to_adj(graph, self_loop=False):
    """
    self_loop: manually add self loop or not
    """
    if graph.A is not None:
        return graph.A
    node_num = graph.node_num
    adj = sp.csr_matrix((graph.adj_wgt, graph.adj_list, graph.adj_idx), shape=(node_num, node_num), dtype=np.float32)
    graph.A = adj
    # i_arr = []
    # j_arr = []
    # data_arr = []
    # for i in range(0, node_num):
    #     for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
    #         i_arr.append(i)
    #         j_arr.append(graph.adj_list[neigh_idx])
    #         data_arr.append(graph.adj_wgt[neigh_idx])
    # adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    # if self_loop:
    #     adj = adj + sp.eye(adj.shape[0])
    return adj

def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))

class Timer:
    """
    time measurement
    """
    def __init__(self, ident=0, logger=None):
        self.count = 0
        self.logger = logger
        self.ident_str = '\t' * ident
        self.restart(coldStart=True)

    def restart(self, name=None, title=None, coldStart=False):
        now = time.time()
        all_time = 0
        if not coldStart:
            self.printIntervalTime(name=title)
            all_time = now - self.startTime
            msg = "%s| Time for this section (%s): %.5f s"%(self.ident_str, name, all_time)
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
        self.startTime = now
        self.prev_time = now
        self.count = 0
        return all_time

    def printIntervalTime(self, name=None):
        now = time.time()
        msg = "%s\t| Interval %d (%s) time %.5f s"%(self.ident_str, self.count, name, now - self.prev_time)
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        self.prev_time = now
        self.count += 1


"""
Data loader for link prediction.
Adapted from and also used by FairAdj
"""
cora_label = {
    "Genetic_Algorithms": 0,
    "Reinforcement_Learning": 1,
    "Neural_Networks": 2,
    "Rule_Learning": 3,
    "Case_Based": 4,
    "Theory": 5,
    "Probabilistic_Methods": 6,
}


def cora(feat_path="datasets/raw/cora/cora.content", edge_path="datasets/raw/cora/cora.cites", scale=True,
         test_ratio=0.1, seed=20) -> Tuple:
    idx_features_labels = np.genfromtxt(feat_path, dtype=np.dtype(str))
    idx_features_labels = idx_features_labels[idx_features_labels[:, 0].astype(np.int32).argsort()]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    nodelist = {idx: node for idx, node in enumerate(idx)}
    X = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    sensitive = np.array(list(map(cora_label.get, idx_features_labels[:, -1])))

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    G = nx.read_edgelist(edge_path, nodetype=int)
    G, test_edges_true, test_edges_false, train_edges_true, train_edges_false = build_test(G, nodelist, test_ratio,
                                                      seed=seed)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

    return G, adj, X, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, nodelist


def citeseer(data_dir="datasets/raw/citeseer", scale=True, test_ratio=0.1, seed=20) -> Tuple:
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open(os.path.join(data_dir, "ind.citeseer.{}".format(names[i])), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    X = sp.vstack((allx, tx)).toarray()
    sensitive = sp.vstack((ally, ty))
    sensitive = np.where(sensitive.toarray() == 1)[1]

    G = nx.from_dict_of_lists(graph)
    test_idx_reorder = parse_index_file(os.path.join(data_dir, "ind.citeseer.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    missing_idx = set(range(min(test_idx_range), max(test_idx_range) + 1)) - set(test_idx_range)
    for idx in missing_idx:
        G.remove_node(idx)

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    nodes = sorted(G.nodes())
    nodelist = {idx: node for idx, node in zip(range(G.number_of_nodes()), list(nodes))}

    G, test_edges_true, test_edges_false, train_edges_true, train_edges_false = build_test(G, nodelist, test_ratio,
                                                      seed=seed)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    adj = nx.adjacency_matrix(G, nodelist=nodes)

    return G, adj, X, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, nodelist


def pubmed(data_dir='datasets/raw/pubmed', scale=True, test_ratio=0.1, seed=20) -> Tuple:
    content = open(os.path.join(data_dir, 'Pubmed-Diabetes.NODE.paper.tab'))
    # skip two header lines
    content.readline()
    content.readline()
    indexes, attributes = {}, []
    feature_list, feat_index, feat_count = [], {}, 0
    for i, line in enumerate(content):
        line = line.strip().split()
        indexes[line[0]] = i

        attr = np.zeros(3)
        attr[int(line[1].split('=')[1]) - 1] = 1
        attributes.append(attr)

        feats = {}
        for f in line[2:-1]:
            feat_label, val = f.split('=')
            if feat_label not in feat_index:
                feat_index[feat_label] = feat_count
                feat_count += 1
            feats[feat_label] = val
        feature_list.append(feats)

    features, attributes = np.zeros((len(feature_list), feat_count)), np.array(attributes)
    attributes = np.where(attributes == 1)[1]
    for i, feats in enumerate(feature_list):
        for key, val in feats.items():
            features[i][feat_index[key]] = val

    edge_list = open(os.path.join(data_dir, 'Pubmed-Diabetes.DIRECTED.cites.tab'))
    # skip two header lines
    edge_list.readline()
    edge_list.readline()
    edges = np.zeros((len(features), len(features)))
    for line in edge_list:
        line = line.strip().split()
        p1, p2 = line[1].split(':')[1], line[3].split(':')[1]
        edges[indexes[p1], indexes[p2]] = 1

    if scale:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    # build data needed by FairAdj
    G = nx.from_numpy_matrix(edges)
    G, test_edges_true, test_edges_false, train_edges_true, train_edges_false = build_test(G,
                                                                                           {i: i for i in range(len(features))},
                                                                                           test_ratio,
                                                                                           seed=seed)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    adj = nx.adjacency_matrix(G)

    return G, adj, features, attributes, train_edges_true, train_edges_false, test_edges_true, test_edges_false, None


def pokec(data_dir='datasets/pokec-n', scale=True, test_ratio=0.1, seed=20) -> Tuple:
    features = np.load(os.path.join(data_dir, 'features.npy'))
    adj = sp.load_npz(os.path.join(data_dir, 'edges.npz'))
    sensitive = np.load(os.path.join(data_dir, 'sens.npy'))

    if scale:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    G = nx.from_scipy_sparse_matrix(adj)
    G, test_edges_true, test_edges_false, train_edges_true, train_edges_false = build_test(G,
                                                                                           {i: i for i in
                                                                                            range(len(features))},
                                                                                           test_ratio,
                                                                                           seed=seed)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    adj = nx.adjacency_matrix(G)

    return G, adj, features, sensitive, test_edges_true, test_edges_false, train_edges_true, train_edges_false, None


def build_test(G: nx.Graph, nodelist: Dict, ratio: float, seed=20) -> Tuple:
    """
    Split training and testing set for link prediction in graph G.
    :param G: nx.Graph
    :param nodelist: idx -> node_id in nx.Graph
    :param ratio: ratio of positive links that used for testing
    :return: Graph that remove all test edges, list of index for test edges
    """

    edges = list(G.edges.data(default=False))
    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    num_test = int(np.floor(num_edges * ratio))
    test_edges_true = []
    test_edges_false = []


    # generate false links for testing
    np.random.seed(seed)
    while len(test_edges_false) < num_test:
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
            continue
        if (idx_u, idx_v) in test_edges_false:
            continue
        else:
            test_edges_false.append((idx_u, idx_v))

    # generate true links for testing
    all_edges_idx = list(range(num_edges))
    np.random.shuffle(all_edges_idx)
    # test_edges_true_idx = all_edges_idx[:num_test]
    test_idx = 0
    while len(test_edges_true) < num_test:
        u, v, _ = edges[all_edges_idx[test_idx]]
        test_idx += 1
        if G.degree[u] <= 1 or G.degree[v] <= 1:
            continue
        G.remove_edge(u, v)
        test_edges_true.append((get_key(nodelist, u), get_key(nodelist, v)))

    # added for logistic regression
    train_edges_true = [(get_key(nodelist, u), get_key(nodelist, v)) for u, v, _ in list(G.edges.data(default=False))]
    train_edges_false = []
    while len(train_edges_false) < len(train_edges_true):
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
            continue
        if (idx_u, idx_v) in train_edges_false or (idx_u, idx_v) in test_edges_false or (idx_u, idx_v) in test_edges_true:
            continue
        else:
            train_edges_false.append((idx_u, idx_v))

    return G, test_edges_true, test_edges_false, train_edges_true, train_edges_false


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value][0]


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

