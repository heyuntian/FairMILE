from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn import metrics
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import os
from typing import Sequence, Tuple, List
from scipy import stats


def summarize_eval_result(ctrl, metrics_dict, sens_num=0, no_sample=False, binary=True):
    res = "************************************************************" + "\n"
    res += "Dataset    :\t" + ctrl.dataset + "\n"
    if ctrl.baseline is False:
        res += "Basic Embed:\t" + ctrl.basic_embed + "\n"
        res += "Refine type:\t" + ctrl.refine_type + "\n"
        res += "Coarsen level:\t" + str(ctrl.coarsen_level) + "\n"
    else:
        res += "Basic Embed:\t" + ctrl.basic_embed + "\n"
        res += "Baseline:\t" + str(ctrl.baseline) + "\n"


    all_keys = sorted(metrics_dict.keys())
    if 'micro-f1' in all_keys:  # particular orders easier to see.
        all_keys = ['auroc']
        if binary:
            all_keys.append('binary-f1')
        all_keys += ['micro-f1', 'macro-f1', 'weighted-f1']
        if not no_sample:
            all_keys.append('samples-f1')
        if sens_num > 0:
            for attr_i in range(sens_num):
                all_keys.append(f'dp-attr_{attr_i}')
            for attr_i in range(sens_num):
                if f'eo-attr_{attr_i}' in metrics_dict:
                    all_keys.append(f'eo-attr_{attr_i}')
    for key in all_keys:
        res += key + ":\t" + metrics_dict[key] + "\n"
    res += "Consumed time:\t" + "{0:.3f}".format(ctrl.embed_time) + " seconds" + "\n"
    res += "************************************************************" + "\n"
    return res


# def eval_multilabel_clf(ctrl, embeddings, truth):
#     attributes = embeddings
#     ctrl.logger.info("Attributes shape: " + str(attributes.shape))
#     ctrl.logger.info("Truth shape: " + str(truth.shape))
#     rnd_time = 10
#     test_size = 0.1
#     metrics_dict = {'micro': [], 'macro': [], 'weighted': [], 'samples': []}
#     sample_num = attributes.shape[0]
#
#     for itr in range(rnd_time):
#         X_train, X_test, y_train, y_test = train_test_split(attributes, truth, test_size=test_size,
#                                                             random_state=np.random.randint(0, 1000))
#         idx = np.random.permutation(sample_num)
#         test_idx = idx[:sample_num * test_size]
#         train_idx = idx[sample_num * test_size:]
#         X_train, X_test, y_train, y_test = attributes[train_idx], attributes[test_idx], truth[train_idx], truth[
#             test_idx]
#         clf = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)  # for multilabel scenario. #penalty='l2'
#         clf.fit(X_train, y_train)
#         y_pred_proba = clf.predict_proba(X_test)
#         y_pred = []
#         for inst in range(len(X_test)):
#             # assume it has the same number of labels as the truth. Same strtegy is used in DeepWalk and Node2Vec paper.
#             y_pred.append(y_pred_proba[inst, :].argsort()[::-1][:sum(y_test[inst, :])])
#
#         y_pred = MultiLabelBinarizer(range(y_pred_proba.shape[1])).fit_transform(y_pred)
#         for key in metrics_dict.keys():
#             metrics_dict[key].append(f1_score(y_test, y_pred, average=key))
#
#     return summarize_eval_result(ctrl, {key + '-f1': "{0:.3f}".format(np.mean(metrics_dict[key])) for key in
#                                         metrics_dict.keys()})


# def eval_oneclass_clf(ctrl, embeddings, truth, sens, sens_num, sens_dim, attr_range, fold=10, no_sample=False):
#     attributes = embeddings
#     ctrl.logger.info("Attributes shape: " + str(attributes.shape))
#     ctrl.logger.info("Truth shape: " + str(truth.shape))
#     # truth = np.argmax(truth, axis=1)
#     truth = truth.flatten()
#     np.random.seed(20)
#     rnd_time = fold
#     train_size = 0.5
#     test_size = 0.25
#     metrics_dict = {'micro-f1': [], 'macro-f1': [], 'weighted-f1': []}
#     if not no_sample:
#         metrics_dict['samples-f1'] = []
#     for attr_i in range(sens_num):
#         metrics_dict[f'dp-attr_{attr_i}'] = []
#     for attr_i in range(sens_num):
#         if attr_range[attr_i+1] - attr_range[attr_i] == 2:
#             metrics_dict[f'eo-attr_{attr_i}'] = []
#     sample_num = attributes.shape[0]
#
#     pos_idx = np.argwhere(sens[:, 0] == 1).flatten()
#     neg_idx = np.argwhere(sens[:, 0] == 0).flatten()
#     pos_num, neg_num = len(pos_idx), len(neg_idx)
#     print(pos_num, neg_num)
#
#     for i in range(rnd_time):
#         # X_train, X_test, y_train, y_test = train_test_split(attributes, truth, test_size=test_size,
#         #                                                     random_state=np.random.randint(0, 1000))
#
#         # idx = np.random.permutation(sample_num)
#         # test_idx = idx[:int(sample_num * test_size)]
#         # train_idx = idx[int(sample_num * test_size):]
#
#         np.random.shuffle(pos_idx)
#         np.random.shuffle(neg_idx)
#         train_idx = np.concatenate([pos_idx[:int(pos_num * train_size)], neg_idx[:int(neg_num * train_size)]])
#         test_idx = np.concatenate([pos_idx[-int(pos_num * test_size):], neg_idx[-int(neg_num * test_size):]])
#
#
#         X_train, X_test, y_train, y_test = attributes[train_idx], attributes[test_idx], \
#                                            truth[train_idx], truth[test_idx]
#
#         clf = LogisticRegression(penalty='l2')
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#
#         # np.save('testid.npy',test_idx)
#         # np.save('y_pred.npy', y_pred)
#         # np.save('y_test.npy', y_test)
#         # np.save('sens_test.npy', sens[test_idx])
#
#         for key in metrics_dict.keys():
#             if key[-3:] == '-f1':
#                 metrics_dict[key].append(f1_score(y_test, y_pred, average=key[:-3]))
#             elif key[:3] == 'dp-':
#                 metrics_dict[key].append(
#                     eval_demographic_parity(int(key[8:]), attr_range, sens[test_idx], y_pred))
#             elif key[:3] == 'eo-':
#                 metrics_dict[key].append(
#                     eval_equal_opportunity(int(key[8:]), attr_range, sens[test_idx], y_pred, y_test)
#                 )
#
#     return summarize_eval_result(ctrl,
#                                  {key: "{0:.3f} +- {1:.3f}".format(np.mean(metrics_dict[key]), np.std(metrics_dict[key])) for key in metrics_dict.keys()}, sens_num=sens_num,
#                                  no_sample=no_sample), \
#            {key: [np.mean(metrics_dict[key]), np.std(metrics_dict[key])] for key in metrics_dict.keys()}


def eval_oneclass_clf(ctrl, embeddings, labels, sens, sens_num, sens_dim, attr_range, seed=20):
    attributes = embeddings
    ctrl.logger.info("Attributes shape: " + str(attributes.shape))
    ctrl.logger.info("Truth shape: " + str(labels.shape))


    metrics_dict = {'auroc': [], 'binary-f1': [], 'micro-f1': [], 'macro-f1': [], 'weighted-f1': []}
    # if not no_sample:
    #     metrics_dict['samples-f1'] = []
    for attr_i in range(sens_num):
        metrics_dict[f'dp-attr_{attr_i}'] = []
    for attr_i in range(sens_num):
        if attr_range[attr_i+1] - attr_range[attr_i] == 2:
            metrics_dict[f'eo-attr_{attr_i}'] = []


    '''
    idx shuffling and selection is copied from NIFTY.
    '''
    train_size = 0.5
    test_size = 0.25
    import random
    random.seed(seed)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    train_idx = np.append(label_idx_0[:int(train_size * len(label_idx_0))],
                          label_idx_1[:int(train_size * len(label_idx_1))])
    # idx_val = np.append(label_idx_0[int(train_size * len(label_idx_0)):int((train_size + test_size) * len(label_idx_0))],
    #                     label_idx_1[int(train_size * len(label_idx_1)):int((train_size + test_size) * len(label_idx_1))])
    test_idx = np.append(label_idx_0[int((train_size + test_size) * len(label_idx_0)):], label_idx_1[int((train_size + test_size) * len(label_idx_1)):])
    # train_idx = all_idx[train_idx]
    # test_idx = all_idx[test_idx]

    # print(train_idx.shape, test_idx.shape, type(train_idx))
    # print(train_idx[:10], test_idx[:10])

    # np.random.seed(seed)
    # pos_idx = np.argwhere(sens[:, 0] == 1).flatten()
    # neg_idx = np.argwhere(sens[:, 0] == 0).flatten()
    # pos_num, neg_num = len(pos_idx), len(neg_idx)
    # print(pos_num, neg_num)
    # np.random.shuffle(pos_idx)
    # np.random.shuffle(neg_idx)
    # train_idx = np.concatenate([pos_idx[:int(pos_num * train_size)], neg_idx[:int(neg_num * train_size)]])
    # test_idx = np.concatenate([pos_idx[-int(pos_num * test_size):], neg_idx[-int(neg_num * test_size):]])


    X_train, X_test, y_train, y_test = attributes[train_idx], attributes[test_idx], \
                                       labels[train_idx].flatten(), labels[test_idx].flatten()

    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    output = clf.predict_proba(X_test)[:, 1]

    # np.save('testid.npy',test_idx)
    # np.save('y_pred.npy', y_pred)
    # np.save('y_test.npy', y_test)
    # np.save('sens_test.npy', sens[test_idx])

    for key in metrics_dict.keys():
        if key[-3:] == '-f1':
            metrics_dict[key].append(f1_score(y_test, y_pred, average=key[:-3]))
        elif key[:3] == 'dp-':
            metrics_dict[key].append(
                eval_demographic_parity(int(key[8:]), attr_range, sens[test_idx], y_pred))
        elif key[:3] == 'eo-':
            metrics_dict[key].append(
                eval_equal_opportunity(int(key[8:]), attr_range, sens[test_idx], y_pred, y_test)
            )
        elif key == 'auroc':
            metrics_dict[key].append(
                roc_auc_score(labels[test_idx], output)
            )

    return summarize_eval_result(ctrl,
                                 {key: "{0:.3f} +- {1:.3f}".format(np.mean(metrics_dict[key]), np.std(metrics_dict[key])) for key in metrics_dict.keys()},
                                 sens_num=sens_num,
                                 no_sample=True), \
           {key: [np.mean(metrics_dict[key]), np.std(metrics_dict[key])] for key in metrics_dict.keys()}


def eval_multilabel_clf(ctrl, embeddings, labels, sens, sens_num, sens_dim, attr_range, seed=20):
    attributes = embeddings
    ctrl.logger.info("Attributes shape: " + str(attributes.shape))
    ctrl.logger.info("Truth shape: " + str(labels.shape))

    metrics_dict = {'auroc': [], 'micro-f1': [], 'macro-f1': [], 'weighted-f1': []}
    for attr_i in range(sens_num):
        metrics_dict[f'dp-attr_{attr_i}'] = []
        metrics_dict[f'eo-attr_{attr_i}'] = []

    train_size = 0.5
    test_size = 0.25
    import random
    random.seed(seed)
    label_unique_values = sorted(np.unique(labels))
    print(label_unique_values)
    num_labels = len(label_unique_values)
    label_idx = [np.where(labels == i)[0] for i in label_unique_values]
    for i in range(num_labels):
        random.shuffle(label_idx[i])
    train_idx = np.concatenate([label_idx_x[:int(train_size * len(label_idx_x))] for label_idx_x in label_idx])
    test_idx = np.concatenate([label_idx_x[int((train_size + test_size) * len(label_idx_x)):] for label_idx_x in label_idx])

    X_train, X_test, y_train, y_test = attributes[train_idx], attributes[test_idx], \
                                       labels[train_idx].flatten(), labels[test_idx].flatten()

    clf = LogisticRegression(multi_class='ovr', penalty='l2', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1).flatten()

    for key in metrics_dict.keys():
        if key[-3:] == '-f1':
            metrics_dict[key].append(f1_score(y_test, y_pred, average=key[:-3]))
        elif key == 'auroc':
            metrics_dict[key].append(
                roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            )
        elif key[:3] == 'dp-':
            metrics_dict[key].append(
                eval_demographic_parity_multiclass(int(key[8:]), attr_range, sens[test_idx], y_pred, label_unique_values))
        elif key[:3] == 'eo-':
            metrics_dict[key].append(
                eval_equal_opportunity_multiclass(int(key[8:]), attr_range, sens[test_idx], y_pred, y_test, label_unique_values)
            )

    return summarize_eval_result(ctrl,
                                 {key: "{0:.3f} +- {1:.3f}".format(np.mean(metrics_dict[key]), np.std(metrics_dict[key])) for key in metrics_dict.keys()},
                                 sens_num=sens_num,
                                 no_sample=True,
                                 binary=False), \
           {key: [np.mean(metrics_dict[key]), np.std(metrics_dict[key])] for key in metrics_dict.keys()}


def eval_demographic_parity(i, attr_range, s_test, y_pred, pos_label=1):
    all_dp = np.zeros(attr_range[i + 1] - attr_range[i])
    for j in range(attr_range[i + 1] - attr_range[i]):
        pos_a = ((s_test[:, i] == j) & (y_pred == pos_label)).sum() / (s_test[:, i] == j).sum()
        all_dp[j] = pos_a
    print(all_dp)
    dp = np.std(all_dp)
    return dp


def eval_demographic_parity_multiclass(i, attr_range, s_test, y_pred, label_values):
    return np.mean([eval_demographic_parity(i, attr_range, s_test, y_pred, pos_label=j) for j in label_values[1:]])


def eval_equal_opportunity(i, attr_range, s_test, y_pred, y_test, pos_label=1):
    all_op = np.zeros(attr_range[i + 1] - attr_range[i])
    assert attr_range[i + 1] - attr_range[i] == 2
    for j in range(attr_range[i + 1] - attr_range[i]):
        pos_a = ((s_test[:, i] == j) & (y_pred == pos_label) & (y_test == pos_label)).sum() / ((s_test[:, i] == j) & (y_test == pos_label)).sum()
        all_op[j] = pos_a
    # eo = abs(all_op[0] - all_op[1])
    eo = np.std(all_op)
    return eo


def eval_equal_opportunity_multiclass(i, attr_range, s_test, y_pred, y_test, label_values):
    return np.mean([eval_equal_opportunity(i, attr_range, s_test, y_pred, y_test, pos_label=j) for j in label_values[1:]])


'''
Evaluation for link prediction
Adapted and also used by FairAdj
'''
THRE = 0.5


def fair_link_eval(
        emb: np.ndarray,
        sensitive: np.ndarray,
        test_edges_true: Sequence[Tuple[int, int]],
        test_edges_false: Sequence[Tuple[int, int]],
        rec_ratio: List[float] = None,
) -> Sequence[List]:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    adj_rec = np.array(np.dot(emb, emb.T), dtype=np.float128)

    preds_pos_intra = []
    preds_pos_inter = []
    for e in test_edges_true:
        # print(sensitive[e[0]], sensitive[e[1]], adj_rec[e[0], e[1]], sigmoid(adj_rec[e[0], e[1]]))
        if sensitive[e[0]] == sensitive[e[1]]:
            preds_pos_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_pos_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg_intra = []
    preds_neg_inter = []
    for e in test_edges_false:
        # print(sensitive[e[0]], sensitive[e[1]], adj_rec[e[0], e[1]], sigmoid(adj_rec[e[0], e[1]]))
        if sensitive[e[0]] == sensitive[e[1]]:
            preds_neg_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_neg_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    res = {}
    for preds_pos, preds_neg, type in zip((preds_pos_intra, preds_pos_inter, preds_pos_intra + preds_pos_inter),
                                          (preds_neg_intra, preds_neg_inter, preds_neg_intra + preds_neg_inter),
                                          ("intra", "inter", "overall")):
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        print('Rate of positive preds:', np.sum(list(map(lambda x: x >= THRE, preds_all))), len(preds_all))
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        # acc_score = accuracy_score(labels_all, list(map(lambda x: x >= THRE, preds_all)))
        err = (np.sum(list(map(lambda x: x >= THRE, preds_pos))) + np.sum(
            list(map(lambda x: x < THRE, preds_neg)))) / (len(preds_pos) + len(preds_neg))

        score_avg = (sum(preds_pos) + sum(preds_neg)) / (len(preds_pos) + len(preds_neg))
        pos_avg, neg_avg = sum(preds_pos) / len(preds_pos), sum(preds_neg) / len(preds_neg)

        res[type] = [roc_score, ap_score, err, score_avg, pos_avg, neg_avg]

    ks_pos = stats.ks_2samp(preds_pos_intra, preds_pos_inter)[0]
    ks_neg = stats.ks_2samp(preds_neg_intra, preds_neg_inter)[0]

    standard = res["overall"][0:3] + [abs(res["intra"][i] - res["inter"][i]) for i in range(3, 6)] + [ks_pos, ks_neg]
    col = ["AUROC", "AP", "Acc", "DP", "EO", "false", "fnr", "tnr"]
    entry = {i: float(j) for i, j in zip(col, standard)}
    return entry


def lp_logistic_eval(
        emb: np.ndarray,
        sensitive: np.ndarray,
        train_edges_true: Sequence[Tuple[int, int]],
        train_edges_false: Sequence[Tuple[int, int]],
        test_edges_true: Sequence[Tuple[int, int]],
        test_edges_false: Sequence[Tuple[int, int]]
):
    y_train = np.array([1] * len(train_edges_true) + [0] * len(train_edges_false))
    X_train = [emb[u] * emb[v] for u, v in (train_edges_true + train_edges_false)]
    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, y_train)

    X_test = [emb[u] * emb[v] for u, v in (test_edges_true + test_edges_false)]
    y_test = np.array([1] * len(test_edges_true) + [0] * len(test_edges_false))
    y_pred = clf.predict(X_test)
    output = clf.predict_proba(X_test)[:, 1]

    roc_score = roc_auc_score(y_test, output)
    ap_score = average_precision_score(y_test, output)
    acc_score = accuracy_score(y_test, y_pred)
    metrics = ["AUROC", "AP", "Acc"]
    results = [roc_score, ap_score, acc_score]

    num_sens = sensitive.shape[1]
    for j in range(num_sens):
        s_test = np.array([(1 if sensitive[u, j] == sensitive[v, j] else 0) for u, v in (test_edges_true + test_edges_false)])
        pos_rate_intra = ((s_test == 1) & (y_pred == 1)).sum() / (s_test == 1).sum()
        pos_rate_inter = ((s_test == 0) & (y_pred == 1)).sum() / (s_test == 0).sum()
        dp = abs(pos_rate_intra - pos_rate_inter)
        pos_expect_intra = sum([a * b for a, b in zip(output, s_test)]) / (s_test == 1).sum()
        pos_expect_inter = sum([a * (1 - b) for a, b in zip(output, s_test)]) / (s_test == 0).sum()
        dp_expect = abs(pos_expect_intra - pos_expect_inter)

        tpr_intra = ((s_test == 1) & (y_pred == 1) & (y_test == 1)).sum() / ((s_test == 1) & (y_test == 1)).sum()
        tpr_inter = ((s_test == 0) & (y_pred == 1) & (y_test == 1)).sum() / ((s_test == 0) & (y_test == 1)).sum()
        eo = abs(tpr_intra - tpr_inter)
        tpr_expect_intra = sum([a * b * c for a, b, c in zip(output, s_test, y_test)]) / ((s_test == 1) & (y_test == 1)).sum()
        tpr_expect_inter = sum([a * (1 - b) * c for a, b, c in zip(output, s_test, y_test)]) / (
                    (s_test == 0) & (y_test == 1)).sum()
        eo_expect = abs(tpr_expect_intra - tpr_expect_inter)

        results.extend([dp, eo, dp_expect, eo_expect])
        metrics.extend([f"DP-attr{j}", f"EO-attr{j}", f"DP_exp-attr{j}", f"EO_exp-attr{j}"])


    entry = {i: float(j) for i, j in zip(metrics, results)}
    return entry
