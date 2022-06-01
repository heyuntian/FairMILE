# This is a wrapper of the official implementation of FairWalk
# GitHub Repository: https://github.com/EnderGed/Fairwalk

import numpy as np
import networkx as nx
import pandas as pd
import multiprocessing as mp
from gensim.models import KeyedVectors, word2vec
import os
import time


def fairwalk(ctrl, graph, walk_len=80, num_walk=20, lr=0.025, epoch=5):
    # Initialize
    node_num = graph.node_num
    attr_id = ctrl.fairwalk.attr_id
    path = "./base_embed_methods/FairWalk"
    rid = int(time.time()) % 10000
    edge_path = export_edgelist(graph, path, rid, attr_id)
    np.random.seed(ctrl.seed)

    # Sample random walks
    run_on_windows = os.name == 'nt'
    walk_path = edge_path[:-8] + "walk" # path + "/tmp/" + "773_0_6740_0.walk"
    print("walk_path %s"%walk_path)
    if run_on_windows:
        os.system('.\\base_embed_methods\\FairWalk\\fast-random-walk\\walk --if={} --of={} --length={} --walks={} -w'.format(edge_path, walk_path, walk_len, num_walk))
    else:
        os.system('./base_embed_methods/FairWalk/fast-random-walk/walk --if={} --of={} --length={} --walks={} -w'.format(edge_path, walk_path, walk_len, num_walk))

    # read walks and process for gensim.model.word2vec
    walks = pd.read_csv(walk_path, header=None)
    walks = walks.loc[np.random.permutation(len(walks))]
    walks = walks.reset_index(drop=True)
    walks = walks.applymap(str)  # gensim only accept list of strings
    # remove the temporary files of edges and walks
    if run_on_windows:
        os.system('del {}'.format(edge_path.replace('/', '\\')))
        os.system('del {}'.format(walk_path.replace('/', '\\')))
    else:
        os.system('rm {}.*'.format(edge_path[:-9]))

    # walks = generate_walks(edge_path)  # deprecated: using our own code for generating random walks.

    # Learn embeddings
    embeddings = emb_train(node_num, walks,
                           walk_len=walk_len,
                           walk_times=num_walk,
                           num_features=ctrl.embed_dim,
                           learning_rate=lr,
                           epoch=epoch,
                           seed=ctrl.seed)

    return embeddings


# export the edgelist file needed by EnderGed's implementation
def export_edgelist(g, path, rid, attr_id):
    node_num = g.node_num
    attr_dist = g.attr_dist
    edge_path = '{}/tmp/fairwalk_{}_{}_{}.edgelist'.format(path, node_num, rid, attr_id)  # do not remove the 'tmp' folder

    attr = np.zeros(node_num, dtype=np.int32)
    for i in range(g.attr_range[attr_id], g.attr_range[attr_id + 1]):
        indices = (attr_dist[:, i] == 1)
        attr[indices] = i - g.attr_range[attr_id]
    all_value_attr = np.unique(attr)
    all_value_attr = [str(ele) for ele in all_value_attr]
    no_attr = len(all_value_attr)
    # adapted from fair.py, line 14, equal_walk_prep
    with open(edge_path, 'w') as f:
        for node in range(node_num):
            neighbors = list(g.get_neighs(node))
            attrs = [str(attr[n]) for n in neighbors]  # convert to str for str-type attributes
            attr_count = [attrs.count(i) for i in all_value_attr]
            # non zero product
            product = np.prod([i for i in attr_count if i != 0])
            attr_weight = {all_value_attr[i]: int(product / attr_count[i]) if attr_count[i] != 0 else 0 for i in range(no_attr)}
            for neigh in neighbors:
                f.write('{},{},{}\n'.format(node, neigh, attr_weight[str(attr[neigh])]))
    return edge_path

# def generate_walks(edge_path, walk_len=80, walk_time=20):
#     def rnd_walk_workers(graph, permuted_idx, proc_begin, proc_end, return_dict, lock):
#         ''' workers to generate random walks.'''
#         all_paths = []
#         for _ in range(walk_time):
#             for start_idx in permuted_idx[proc_begin: proc_end]:
#                 path = [start_idx]
#                 for _ in range(walk_len):
#                     curr_idx = path[-1]
#                     t = list(graph.edges(curr_idx, data='weight'))
#                     neigh = [b for (a, b, c) in t]
#                     wgts = np.array([c for (a, b, c) in t])
#                     path.append(np.random.choice(neigh, p=wgts / float(sum(wgts))))
#                 all_paths.append(map(str, path))
#         with lock:
#             return_dict[proc_begin] = all_paths
#
#     # Multi-processing
#     manager = mp.Manager()
#     workers = mp.cpu_count()
#     return_dict = manager.dict()
#     lock = mp.Lock()
#     jobs = []
#
#     # Graph
#     G = nx.read_weighted_edgelist(edge_path, delimiter=',', nodetype=int)
#     n = G.number_of_nodes()
#
#     chunk_size = n // workers
#     permuted_idx = np.random.permutation(n)
#     for i in range(workers):
#         proc_begin = i * chunk_size
#         proc_end = (i + 1) * chunk_size
#         if i == workers - 1:
#             proc_end = n
#         p = mp.Process(target=rnd_walk_workers, args=(G, permuted_idx, proc_begin, proc_end, return_dict, lock))
#         jobs.append(p)
#     for p in jobs:
#         p.start()
#     for proc in jobs:
#         proc.join()
#     all_paths = []
#     key_arr = sorted(return_dict.keys())
#     np.random.shuffle(key_arr)
#     for key in key_arr:
#         all_paths += return_dict[key]
#     return all_paths


# adapted from emb.py, line 41, emb_train
def emb_train(n, walks, walk_len=80, walk_times=20, num_features=128, learning_rate=0.025, epoch=5, seed=20):
    min_word_count = 0
    num_workers = mp.cpu_count()
    context = 10
    downsampling = 1e-3

    # gensim does not support numpy array, thus, walks.tolist()
    walks = walks.groupby(0).head(walk_times).values[:, :walk_len].tolist()
    emb = word2vec.Word2Vec(walks, \
                            sg=1, \
                            workers=num_workers, \
                            vector_size=num_features, min_count=min_word_count, \
                            epochs=epoch, alpha=learning_rate, \
                            window=context, sample=downsampling,
                            seed=seed)
    print('training done')
    embeddings = np.zeros((n, num_features), dtype=np.float32)
    for i in range(n):
        if str(i) in emb.wv:
            embeddings[i] = emb.wv[str(i)]
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    return embeddings