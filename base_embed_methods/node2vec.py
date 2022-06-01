# This is adapted from fairwalk.py

import numpy as np
import networkx as nx
import pandas as pd
import multiprocessing as mp
from gensim.models import KeyedVectors, word2vec
import os


def node2vec(ctrl, graph):
    # Initialize
    node_num = graph.node_num
    path = "./base_embed_methods/FairWalk"
    edge_path = export_edgelist(graph, path)
    np.random.seed(ctrl.seed)

    # Sample random walks
    run_on_windows = os.name == 'nt'
    walk_path = edge_path[:-8] + "walk" # path + "/tmp/" + "773_0_6740_0.walk"
    print("walk_path %s"%walk_path)
    if run_on_windows:
        os.system('.\\base_embed_methods\\FairWalk\\fast-random-walk\\walk --if={} --of={} --length=80 --walks=20 -w'.format(edge_path, walk_path))
    else:
        command = './base_embed_methods/FairWalk/fast-random-walk/walk --if={} --of={} --length=80 --walks=20 -w'.format(edge_path, walk_path)
        ctrl.logger.info(command)
        os.system(command)

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

    # Learn embeddings
    embeddings = emb_train(node_num, walks, num_features=ctrl.embed_dim, seed=ctrl.seed)

    return embeddings


# export the edgelist file needed by EnderGed's implementation
def export_edgelist(g, path):
    node_num = g.node_num
    edge_path = '{}/tmp/node2vec_{}.edgelist'.format(path, node_num)  # do not remove the 'tmp' folder

    # adapted from fair.py, line 14, equal_walk_prep
    with open(edge_path, 'w') as f:
        for node in range(node_num):
            for j in range(g.adj_idx[node], g.adj_idx[node+1]):
                f.write('{},{},{}\n'.format(node, g.adj_list[j], int(g.adj_wgt[j])))
    return edge_path


# adapted from emb.py, line 41, emb_train
def emb_train(n, walks, walk_len=80, walk_times=20, num_features=128, seed=20):
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
                            window=context, sample=downsampling, \
                            seed=seed)
    print('training done')
    embeddings = np.zeros((n, num_features), dtype=np.float32)
    for i in range(n):
        if str(i) in emb.wv:
            embeddings[i] = emb.wv[str(i)]
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    return embeddings