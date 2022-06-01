"""
graph.py: Data structure of the graph

@author: Yuntian He
"""

import numpy as np


class Graph(object):

    def __init__(self, node_num, edge_num, sens_num, sens_dim, attr_range):
        self.node_num = node_num  # number of nodes
        self.edge_num = edge_num  # number of edges
        self.sens_num = sens_num    # number of sensitive attributes
        self.sens_dim = sens_dim    # sum of sensitive attribute values
        self.adj_list = np.zeros(edge_num, dtype=np.int32)  # adjacency list
        self.adj_wgt = np.zeros(edge_num, dtype=np.float32)  # weight of each edge
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # index of the beginning neighbor in adj_list of each# node
        self.node_wgt = np.zeros(node_num, dtype=np.float32)    # weight of each node
        self.degree = np.zeros(node_num, dtype=np.float32)   # sum of incident edges' weights of each node
        self.attr_range = attr_range    # number of values for each sensitive attributes
        self.attr_dist = np.zeros((node_num, sens_dim), dtype=np.float32)
        self.norm_attr_dist = np.zeros((node_num, sens_dim), dtype=np.float32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1

        self.coarser = None
        self.finer = None
        self.C = None
        self.A = None

    def resize_adj(self, edge_num):
        """ Resize the adjacency list/wgts based on the number of edges."""
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        """obtain the list of neigbors given a node."""
        idx_start = self.adj_idx[idx]
        idx_end = self.adj_idx[idx + 1]
        return self.adj_list[idx_start:idx_end]

    def get_neigh_edge_wgts(self, idx):
        """obtain the weights of neighbors given a node."""
        idx_start = self.adj_idx[idx]
        idx_end = self.adj_idx[idx + 1]
        return self.adj_wgt[idx_start:idx_end]
