"""
Model for Refinement.
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from scipy.stats import entropy
from utils import graph_to_adj
import tensorflow as tf
from tensorflow.keras import regularizers
from layers import *



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def fair_graph_adj(ctrl, graph):
    node_num = graph.node_num
    adj_idx = graph.adj_idx
    adj_list = graph.adj_list
    adj_wgt = graph.adj_wgt
    attr_dist = graph.attr_dist
    node_wgt = graph.node_wgt
    sens_num = graph.sens_num

    fair_adj_wgt = np.zeros_like(adj_wgt)
    for idx in range(node_num):
        attr1 = attr_dist[idx]
        wgt1 = node_wgt[idx]
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            fair_adj_wgt[j] = ctrl.attr_sim(ctrl, attr1, attr_dist[neigh], wgt1, node_wgt[neigh], sens_num)

    adj = sp.csr_matrix((fair_adj_wgt, adj_list, adj_idx), shape=(node_num, node_num), dtype=np.float32)
    adj.setdiag(0)
    return normalize_adj(adj)


def fair_graph_adj_binary(ctrl, graph, attr_mtx, threshold=0.5, reverse=True):
    """
    Create a sparse matrix to binary-weight edges in graph in terms of fairness
    :param ctrl:
    :param graph:
    :return:
    """

    node_num = graph.node_num
    adj_idx = graph.adj_idx
    adj_list = graph.adj_list
    adj_wgt = graph.adj_wgt
    attr_dist = attr_mtx
    node_wgt = graph.node_wgt
    sens_num = graph.sens_num

    fair_adj_wgt = np.zeros_like(adj_list)
    for idx in range(node_num):
        attr1 = attr_dist[idx]
        wgt1 = node_wgt[idx]
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if reverse:
                fair_adj_wgt[j] = -1 if ctrl.attr_sim(ctrl, attr1, attr_dist[neigh], wgt1, node_wgt[neigh],
                                                      sens_num) >= threshold else 0
            else:
                fair_adj_wgt[j] = 1 if ctrl.attr_sim(ctrl, attr1, attr_dist[neigh], wgt1, node_wgt[neigh],
                                                  sens_num) < threshold else 0

    adj = sp.csr_matrix((fair_adj_wgt, adj_list, adj_idx), shape=(node_num, node_num), dtype=np.float32)
    adj.setdiag(0)
    return adj


def preprocess_to_gcn_adj(adj, lda):  # D^{-0.5} * A * D^{-0.5} : normalized, symmetric convolution operator.
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    self_loop_wgt = np.array(adj.sum(1)).flatten() * lda  # self loop weight as much as sum. This is part is flexible.
    adj_normalized = normalize_adj(adj + sp.diags(self_loop_wgt)).tocoo()
    return adj_normalized


def convert_sparse_matrix_to_sparse_tensor(X):
    if not sp.isspmatrix_coo(X):
        X = X.tocoo()
    indices = np.mat([X.row, X.col]).transpose()
    return tf.SparseTensor(indices, X.data, X.shape)


def normalize_attr_mtx(attr_mtx, fine_graph=None):
    return softmax(attr_mtx, axis=1)
    # norm_attr_dist = attr_mtx / attr_mtx.sum(axis=1).reshape(-1, 1)
    # return norm_attr_dist


class GCN(tf.keras.Model):
    """
    Normal GCN with no fairness loss.
    """

    def __init__(self, ctrl):
        super().__init__()
        # Utils and hyperparameters
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.act_func = ctrl.refine_model.act_func
        self.wgt_decay = ctrl.refine_model.wgt_decay
        self.regularized = ctrl.refine_model.regularized
        self.learning_rate = ctrl.refine_model.learning_rate
        self.hidden_layer_num = ctrl.refine_model.hidden_layer_num
        self.lda = ctrl.refine_model.lda
        self.epoch = ctrl.refine_model.epoch
        self.early_stopping = ctrl.refine_model.early_stopping
        self.optimizer = ctrl.refine_model.tf_optimizer(learning_rate=self.learning_rate)
        self.lambda_fl = ctrl.refine_model.lambda_fl
        self.ctrl = ctrl

        # Layers
        self.conv_layers = []
        for i in range(self.hidden_layer_num):
            conv = GCNConv(self.embed_dim, activation=self.act_func, use_bias=False,
                           kernel_regularizer=regularizers.l2(l2=self.wgt_decay / 2.0) if self.regularized else None)
            self.conv_layers.append(conv)

    def call(self, gcn_A, input_embed):
        curr = input_embed
        for i in range(self.hidden_layer_num):
            curr = self.conv_layers[i]([gcn_A, curr])
        output = tf.nn.l2_normalize(curr, axis=1)
        return output

    def train(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed
        self.logger.info(f'initial_embed: {initial_embed.shape}')
        self.logger.info(f'fine_embed: {fine_embed.shape}')

        loss_arr = []
        for i in range(self.epoch):
            with tf.GradientTape() as tape:
                pred_embed = self.call(struc_A, initial_embed)
                acc_loss = tf.compat.v1.losses.mean_squared_error(fine_embed,
                                                                  pred_embed) * self.embed_dim  # tf.keras.losses.mean_squared_error(y_true=fine_embed, y_pred=pred_embed) * self.embed_dim
                loss = acc_loss
                # print(f'Epoch {i}, Loss: {loss}, Acc Loss: {acc_loss}')
                loss_arr.append(loss)
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

    def predict(self, coarse_graph=None, fine_graph=None, coarse_embed=None, last_level=False):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        initial_embed = fine_graph.C.dot(coarse_embed)
        return self.call(struc_A, initial_embed)


class GCNNew(tf.keras.Model):
    """
    GCN with fairness loss.
    """

    def __init__(self, ctrl):
        super().__init__()
        # Utils and hyperparameters
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.attr_dim = ctrl.attr_dim
        self.act_func = ctrl.refine_model.act_func
        self.wgt_decay = ctrl.refine_model.wgt_decay
        self.regularized = ctrl.refine_model.regularized
        self.learning_rate = ctrl.refine_model.learning_rate
        self.hidden_layer_num = ctrl.refine_model.hidden_layer_num
        self.lda = ctrl.refine_model.lda
        self.epoch = ctrl.refine_model.epoch
        self.early_stopping = ctrl.refine_model.early_stopping
        self.optimizer = ctrl.refine_model.tf_optimizer(learning_rate=self.learning_rate)
        self.lambda_fl = ctrl.refine_model.lambda_fl
        self.lambda_fl2 = ctrl.refine_model.lambda_fl2
        self.fair_threshold = ctrl.refine_model.fair_threshold
        self.ctrl = ctrl

        # Layers
        self.conv_layers = []
        for i in range(self.hidden_layer_num):
            conv = GCNConv(self.embed_dim, activation=self.act_func, use_bias=False,
                           kernel_regularizer=regularizers.l2(l2=self.wgt_decay / 2.0) if self.regularized else None)
            self.conv_layers.append(conv)

    def call(self, gcn_A, attr_mtx, input_embed):
        curr = input_embed
        for i in range(self.hidden_layer_num):
            curr = self.conv_layers[i]([gcn_A, tf.concat([curr, attr_mtx], axis=1)])
        output = tf.nn.l2_normalize(curr, axis=1)
        return output

    def train(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        """
        Train the model
        :param coarse_graph:
        :param fine_graph:
        :param coarse_embed:
        :param fine_embed:
        :return:
        """

        '''
        Graph data
        adj:  csr_matrix, adjacency matrix
        struc_A:  sparse tensor, normalized adj matrix for message passing in GCN
        use_neg:  bool, use negative sample for the fairness constraint        
        '''
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        norm_attr_dist = normalize_attr_mtx(fine_graph.attr_dist)
        fair_A = fair_graph_adj_binary(self.ctrl, fine_graph, norm_attr_dist,
                                           threshold=self.fair_threshold,
                                           reverse=True)

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed
        self.logger.info(f'initial_embed: {initial_embed.shape}')
        self.logger.info(f'fine_embed: {fine_embed.shape}')

        # early_stopping = self.early_stopping
        loss_arr = []
        for i in range(self.epoch):
            with tf.GradientTape() as tape:
                pred_embed = self.call(struc_A, norm_attr_dist, initial_embed)
                acc_loss = tf.compat.v1.losses.mean_squared_error(fine_embed,
                                                                  pred_embed) * self.embed_dim

                # fair loss 1: penalize neighbors from the same group
                reconstruct_fair = tf.math.sigmoid(tf.linalg.matmul(pred_embed, pred_embed, transpose_b=True))
                fair_loss = tf.math.reduce_sum(tf.math.multiply(reconstruct_fair, fair_A.todense())) / (abs(
                    fair_A.sum()) + 1e-10)

                # fair loss 2: penalize the distance among the groups
                '''
                Group fair loss
                To see the differnce of each group's embedding sum
                Line 1: Simply sum up
                Line 2: Normalize by each group's sum of weights in attr_dist
                '''

                fair_loss_2 = 0

                loss = acc_loss * (1 - self.lambda_fl - self.lambda_fl2) + fair_loss * self.lambda_fl + fair_loss_2 * self.lambda_fl2
                print(f'Epoch {i}, Loss: {loss}, Acc Loss: {acc_loss}, Fair Loss: {fair_loss} + {fair_loss_2}')
                loss_arr.append(loss)
                # if i > early_stopping and loss_arr[-1] > np.mean(loss_arr[-(early_stopping + 1):-1]):
                #     self.logger.info("Early stopping...")
                #     break
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

    def predict(self, coarse_graph=None, fine_graph=None, coarse_embed=None, last_level=False):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        initial_embed = fine_graph.C.dot(coarse_embed)
        norm_attr_dist = normalize_attr_mtx(fine_graph.attr_dist)
        pred_embed = self.call(struc_A, norm_attr_dist, initial_embed)
        return pred_embed



"""
Define your attribute divergence functions here.
You need to manually add them to argument parser in `main.py`. 
"""


def dissimilarity_attr(ctrl, attr1, attr2, wgt1, wgt2, sens_num):
    return attr2.dot(wgt1 - attr1) / sens_num / (wgt1 * wgt2)


def exchange(ctrl, attr1, attr2, wgt1, wgt2, sens_num):
    return 1.0 / (1.0 + attr1.dot(wgt2 - attr2) / sens_num / (wgt1 * wgt2))


def exc_inv(ctrl, attr1, attr2, wgt1, wgt2, sens_num):
    return 1.0 / (1.0 + attr1.dot(wgt2 - attr2))


def abs_diff(ctrl, attr1, attr2, wgt1, wgt2, sens_num):
    return np.abs(attr1 - attr2).sum()


def kl_dvg(ctrl, attr1, attr2, wgt1, wgt2, sens_num):
    """
    Compute the KL-divergence of two normalized attribute vector.
    Note that the attribute vectors must be normalized (sum to 1) before calling it.
    """
    return 1 - 1 / (1 + entropy(attr1, attr2))