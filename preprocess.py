#!/usr/bin/env python
"""
Preprocess the data into a uniform format
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from utils import Mapping
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', default='german',
                        help='Input graph file')
    parser.add_argument('--task', default='nc', choices=['nc', 'lp'],
                        help='Task for evaluation (nc: node classification/ lp: link prediction)')
    parser.add_argument('--output', default='datasets/',
                        help='Path for output')
    args = parser.parse_args()
    return args


def read_nifty(args, pred_attrs, discard_attrs, sens_attrs):
    """
    Read Datasets used by Nifty
    Adapted from GitHub @chirag126/nifty/utils.py, line 174, load_german
    https://github.com/chirag126/nifty/blob/main/utils.py
    """

    path = 'datasets/raw/{}/'.format(args.data)
    output_path = '{}/{}'.format(args.output, args.data)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Basics
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(args.data)))
    header = list(idx_features_labels.columns)
    for attr in pred_attrs:
        header.remove(attr)
    for attr in discard_attrs:
        header.remove(attr)

    # Sensitive attribute: Numpy array
    sens = scale_attr(idx_features_labels, sens_attrs)
    node_num = sens.shape[0]
    np.save(f'{output_path}/sens.npy', sens)
    print(f'Sensitive attributes: {sens.shape}')
    # print(sens[:10])

    # Predict attribute: NumPy array
    labels = scale_attr(idx_features_labels, pred_attrs)
    labels = np.append(np.arange(node_num).reshape(-1, 1), labels, 1).astype(np.int32)
    np.save(f'{output_path}/labels.npy', labels)
    print(f'Labels: {labels.shape}')
    # print(labels[:10])

    # Features: NumPy array
    features = np.array((sp.csr_matrix(idx_features_labels[header], dtype=np.float32)).todense())
    np.save(f'{output_path}/features.npy', features)
    print(f'Normal attributes: {features.shape}')
    # print(features[:5])

    # Graph Structure: CSR matrix
    edges_unordered = np.genfromtxt(f'{path}/{args.data}_edges.txt').astype('int')
    idx = np.arange(node_num)
    old2new = {j: i for i, j in enumerate(idx)}
    new2old = {i: j for i, j in enumerate(idx)}
    mapping = Mapping(old2new, new2old)
    edges = np.array(list(map(old2new.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    csr_edges = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), \
                              shape=(node_num, node_num),
                              dtype=np.float32)
    csr_edges = csr_edges + csr_edges.T.multiply(csr_edges.T > csr_edges) - csr_edges.multiply(csr_edges.T > csr_edges)
    csr_edges = csr_edges + sp.eye(csr_edges.shape[0])
    sp.save_npz(f'{output_path}/edges.npz', csr_edges)
    print(f'edge_num: {csr_edges.count_nonzero()}')


def scale_attr(df, col_list):
    """
    Scale the attributes in col_list to integers starting from 0, then create an array of attributes in col_list
    If there is 1 attribute, it returns an (n, 1) array.
    :param df:
    :param col_list:
    :return: attribute array of (n, len(col_list))
    """
    n = df.shape[0]
    num_attrs = len(col_list)
    for attr_id in range(num_attrs):
        attr = col_list[attr_id]
        uniq_values = list(df[attr].unique())
        flag_not_all_int = False
        flag_has_negative = False
        for i in range(len(uniq_values)):
            is_int = isinstance(uniq_values[i], int) or isinstance(uniq_values[i], np.int64)
            flag_not_all_int = flag_not_all_int or not is_int
            if is_int:
                flag_has_negative = flag_has_negative or (uniq_values[i] < 0)

        if flag_not_all_int or flag_has_negative:
            if flag_not_all_int:
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            else:
                uniq_values = sorted(uniq_values)
                map_attr = {j: i for i, j in enumerate(uniq_values)}
            data = list(map(map_attr.get, df[attr]))
            df[attr] = data
    arr = df[col_list].values
    return arr


def reformat_attr(df, col_list):
    """
    Create a binary array of attributes. If there is 1 attribute with 3 values, it returns an (n, 3) array.

    :param df: dataframe of raw data
    :param col_list: list of columns to be processed
    :return: a binary array of attribute values
    """
    n = df.shape[0]
    arr = None
    for attr in col_list:
        uniq_values = list(df[attr].unique())
        map_sens_attr = {j: i for i, j in enumerate(uniq_values)}
        arr_tmp = np.zeros((n, len(uniq_values)))
        for val in uniq_values:
            idx = df.index[df[attr] == val].tolist()
            arr_tmp[idx, map_sens_attr[val]] = 1
        if arr is None:
            arr = arr_tmp
        else:
            arr = np.hstack((arr, arr_tmp))
    return arr


def read_pokec(args, pred_attrs, discard_attrs, sens_attrs):
    """
    Read Pokec data used by FairGCN (Dai and Wang, WSDM '21)
    Adapted from https://github.com/EnyanDai/FairGNN/blob/main/src/utils.py
    :param args:
    :param pred_attrs:
    :param discard_attrs:
    :param sens_attrs:
    :return:
    """
    path = 'datasets/raw/{}/'.format(args.data)
    output_path = '{}/{}'.format(args.output, args.data)
    sens_attr, predict_attr = sens_attrs[0], pred_attrs[0]

    idx_features_labels = pd.read_csv(os.path.join(path, "region_job.csv"))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)
    for attr in discard_attrs:
        header.remove(attr)

    labels = idx_features_labels[predict_attr].values

    # Sensitive attribute: Numpy array
    sens = scale_attr(idx_features_labels, sens_attrs)
    node_num = sens.shape[0]
    np.save(f'{output_path}/sens.npy', sens)
    print(f'Sensitive attributes: {sens.shape}')
    # print(sens[:10])

    # Predict attribute: NumPy array
    label_idx = np.where(labels >= 0)[0]
    labels = labels[label_idx]
    labels = np.append(label_idx.reshape(-1, 1), labels.reshape(-1, 1), 1).astype(np.int32)
    np.save(os.path.join(output_path, 'labels.npy'), labels)
    print(f'Labels: {labels.shape}')
    # print(labels[:10])

    # Features: NumPy array
    features = np.array((sp.csr_matrix(idx_features_labels[header], dtype=np.float32)).todense())
    np.save(f'{output_path}/features.npy', features)
    print(f'Normal attributes: {features.shape}')
    # print(features[:5])

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "region_job_relationship.txt")).astype('int')
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    sp.save_npz(f'{output_path}/edges.npz', adj)
    print(f'edge_num: {adj.count_nonzero()}')

    return adj, features, labels, sens


if __name__ == '__main__':
    args = parse_args()
    if args.data == 'german':
        pred_attrs = ['GoodCustomer']
        discard_attrs = ['OtherLoansAtStore', 'PurposeOfLoan']
        sens_attrs = ['Gender']
        read_nifty(args, pred_attrs, discard_attrs, sens_attrs)
    if args.data == 'credit':
        pred_attrs = ['NoDefaultNextMonth']
        discard_attrs = ['Single']
        sens_attrs = ['Age']
        read_nifty(args, pred_attrs, discard_attrs, sens_attrs)
    if args.data == 'bail':
        pred_attrs = ['RECID']
        discard_attrs = []
        sens_attrs = ['WHITE']
        read_nifty(args, pred_attrs, discard_attrs, sens_attrs)
    if args.data[:5] == 'pokec':
        pred_attrs = ['I_am_working_in_field']
        discard_attrs = []
        sens_attrs = ['region']
        read_pokec(args, pred_attrs, discard_attrs, sens_attrs)