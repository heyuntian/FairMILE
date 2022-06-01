#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from config import Config
from utils import setup_custom_logger, read_data
from embed import multilevel_embed, call_baselines
from coarsen import general_match, mile_match
from refine_model import GCN, GCNTest, GCNNew
from evaluate import *
import multiprocessing
import tensorflow as tf
import importlib
import json


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--task', required=False, default='nc', choices=['nc', 'lp'],
                        help='Downstream task (nc or lp)')
    parser.add_argument('--data', required=False, default='german',
                        help='Input graph file')
    parser.add_argument('--format', required=False, default='nifty', choices=['nifty', 'edgelist'],
                        help='Format of the input graph file (nifty/edgelist)')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='netmf',
                        choices=['deepwalk', 'node2vec', 'grarep', 'netmf', 'fairwalk'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--baseline', action='store_true',
                        help='Call the method stand-alone.')

    ''' Parameters for Coarsening '''
    parser.add_argument('--coarse-type', required=False, default='fair',
                        choices=['fair', 'normal'],
                        help="Choose MILE's matching method or the new one with fairness.")
    parser.add_argument('--coarsen-level', default=1, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--coarse-fair-func', default='kl_dvg',
                        choices=['mergefair_group', 'abs_diff', 'kl_dvg'],
                        help='Metric for fairness in merging nodes')
    parser.add_argument('--wgt-merge', default=0.5, type=float,
                        help='Weight of fairness term in matching')

    ''' Parameters for Refinement'''
    parser.add_argument('--refine-type', required=False, default='new',
                        choices=['gcn', 'new'],
                        help='The method for refining embeddings.')
    parser.add_argument('--lambda-fl', default=0.5, type=float,
                        help='Weight of the 1st fair loss in training')
    parser.add_argument('--lambda-fl2', default=0, type=float,
                        help='Weight of the 2nd fair loss (if applicable) in training')
    parser.add_argument('--fair-threshold', default=0.5, type=float,
                        help='Threshold for fairness penalization')
    parser.add_argument('--attr-sim', default='kl_dvg',
                        choices=['exchange', 'exc_inv', 'dissimilarity_attr', 'abs_diff', 'kl_dvg'],
                        help='Metric of attribute dissimilarity in refinement')
    parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--epoch', default=200, type=int,
                        help='Epochs for training the refinement model')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    # parser.add_argument('--use-neg', action='store_true',
    #                     help='Number of min-hashes used for SEM.')
    # parser.add_argument('--negative', default=20, type=int,
    #                     help='Number of min-hashes used for SEM.')

    # parser.add_argument('--k-fold', default=5, type=int,
    #                     help='Number of folds for evaluation')
    ''' Storing result and embedding'''
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed.')
    parser.add_argument('--jobid', default=0, type=int,
                        help='slurm job id')
    parser.add_argument('--store-embed', action='store_true',
                        help='Store the embeddings.')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--only-eval', action='store_true',
                        help='Evaluate existing embeddings.')

    args = parser.parse_args()
    return args


def sync_config_args(ctrl, args, graph):
    # General
    ctrl.dataset = args.data
    ctrl.logger = setup_custom_logger('FairEmbed')
    ctrl.embed_dim = args.embed_dim
    ctrl.coarsen_level = args.coarsen_level
    ctrl.seed = args.seed

    # Coarsening
    ctrl.coarse_type = args.coarse_type
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** args.coarsen_level))
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    ctrl.coarse_fair_func = getattr(importlib.import_module("coarsen"), args.coarse_fair_func)
    ctrl.wgt_merge = args.wgt_merge

    # Embedding
    ctrl.basic_embed = args.basic_embed
    ctrl.workers = args.workers

    # Refinement
    ctrl.refine_type = args.refine_type
    ctrl.attr_sim = getattr(importlib.import_module("refine_model"), args.attr_sim)
    ctrl.refine_model.fair_threshold = args.fair_threshold
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.epoch = args.epoch
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.lda = args.self_weight
    ctrl.refine_model.lambda_fl = args.lambda_fl
    ctrl.refine_model.lambda_fl2 = args.lambda_fl2
    # ctrl.refine_model.use_negative = args.use_neg
    # ctrl.refine_model.negative = args.negative

    # Baselines
    ctrl.baseline = args.baseline

    # Evaluation
    ctrl.only_eval = args.only_eval

    ctrl.logger.info(args)


def select_coarse_method(ctrl):
    coarse_method = None
    if ctrl.coarse_type == 'normal':
        coarse_method = mile_match
    elif ctrl.coarse_type == 'fair':
        coarse_method = general_match
    else:
        print('Invalid coarsening method')
        exit(0)
    return coarse_method


def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'new':
        refine_model = GCNNew
    return refine_model


def evaluate_embeddings(ctrl, truth_mat, embeddings, sens, sens_num, sens_dim, attr_range):
    '''
    Evaluation for node classification
    :param ctrl:
    :param truth_mat:
    :param embeddings:
    :param sens:
    :param sens_num:
    :param sens_dim:
    :param attr_range:
    :return:
    '''
    idx_arr = truth_mat[:, 0].reshape(-1)  # this is the original index
    raw_truth = truth_mat[:, 1:]  # multi-class result
    embeddings = embeddings[idx_arr, :]  # in the case of yelp, only a fraction of data contains label.
    # res, entry = eval_oneclass_clf(ctrl, embeddings, truth, sens, sens_num, sens_dim, attr_range, fold=fold, no_sample=truth.shape[1] == 1)
    if len(np.unique(raw_truth)) == 2:
        res, entry = eval_oneclass_clf(ctrl, embeddings, raw_truth, sens, sens_num, sens_dim, attr_range, seed=ctrl.seed)
    else:
        res, entry = eval_multilabel_clf(ctrl, embeddings, raw_truth, sens, sens_num, sens_dim, attr_range, seed=ctrl.seed)
    print(res)
    return entry


def embedding_filename(args, ctrl):
    return f'embeddings/{args.data}_{args.task}_' + \
               ('' if args.baseline else f'FairMILE-{args.coarsen_level}-') + \
               (f'{args.basic_embed}_{ctrl.fairwalk.attr_id}' if args.basic_embed == 'fairwalk' else args.basic_embed) + \
               f'_{args.embed_dim}_{args.wgt_merge}_{args.lambda_fl}_{args.seed}.npy'


def store_embeddings(args, ctrl, embeddings):
    filename = embedding_filename(args, ctrl)
    with open(filename, 'wb') as f:
        np.save(f, embeddings)


if __name__ == '__main__':
    ctrl = Config()
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.task == 'nc':
        graph, features, labels, sens = read_data(args)
    else:
        graph, features, sens, train_edges_true, train_edges_false, test_edges_true, test_edges_false = read_data(args)
    sync_config_args(ctrl, args, graph)

    if ctrl.only_eval:
        ctrl.embed_time = 0
        embeddings = np.load(embedding_filename(args, ctrl))
    else:
        base_embed = select_base_embed(ctrl)
        if ctrl.baseline is False:
            # Select coarsening method and refinement model
            coarse_method = select_coarse_method(ctrl)
            refine_model = select_refine_model(ctrl)

            # generate embeddings
            embeddings = multilevel_embed(ctrl, graph, coarse_method, base_embed, refine_model)
        else:
            embeddings = call_baselines(ctrl, graph, base_embed)

    # evaluate
    entry = None
    if not args.no_eval:
        if args.task == 'nc':
            entry = evaluate_embeddings(ctrl, labels, embeddings, sens, graph.sens_num, graph.sens_dim, graph.attr_range)
            entry['time'] = [ctrl.embed_time]
        else:
            # entry = fair_link_eval(embeddings, sens.flatten(), test_edges_true, test_edges_false)
            entry = lp_logistic_eval(embeddings, sens,
                                     train_edges_true, train_edges_false, test_edges_true, test_edges_false)
            entry['time'] = ctrl.embed_time
            print(entry)

    # write to json
    if entry:
        filename = f'{args.jobid}.json'
        if os.path.exists(filename):
            fr = open(filename)
            jd = json.load(fr)
            fr.close()
        else:
            jd = json.loads("""{"results": []}""")

        entry['dataset'] = args.data
        entry['baseline'] = '*' if args.baseline else f'FairMILE-{args.coarse_type[:3]}-{args.refine_type}'
        entry['method'] = f'{args.basic_embed}-{args.embed_dim}'
        entry['c-level'] = '*' if args.baseline else args.coarsen_level
        entry['parameter'] = '*' if args.baseline \
            else f'{args.wgt_merge}_{args.lambda_fl}'
            # else f'{args.attr_sim}_{args.fair_threshold}_{args.epoch}_{args.learning_rate}_{args.lambda_fl}_{args.lambda_fl2}'
        entry['seed'] = args.seed

        jd['results'].append(entry)
        js = json.dumps(jd, indent=2)

        fw = open(filename, 'w')
        fw.write(js)
        fw.close()

    # Store embeddings
    if args.store_embed:
        store_embeddings(args, ctrl, embeddings)
