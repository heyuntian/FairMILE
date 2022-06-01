"""
Multi-level fair embedding.
"""
import time
from utils import Timer, normalized
from coarsen import general_match, create_coarse_graph
import numpy as np
import tensorflow as tf
from refine_model import GCN, Debias


def print_coarsen_info(ctrl, g):
    cnt = 0
    while g is not None:
        ctrl.logger.info(f'Level {cnt} --- # nodes: {g.node_num} , # edges: {g.edge_num}')
        g = g.coarser
        cnt += 1


def multilevel_embed(ctrl, graph, coarse_method, base_embed, refine_model):
    """
    Multi-level fair embedding method.
    :param ctrl:
    :param graph:
    :param coarse_method:
    :param base_embed:
    :param refine_model:
    :return: embeddings: (n, embed_dim) array
    """

    # Start
    np.random.seed(ctrl.seed)
    tf.random.set_seed(ctrl.seed)
    timer = Timer(logger=ctrl.logger, ident=0)

    # Graph coarsening
    if ctrl.coarsen_level > 0:
        original_graph = graph
        coarsen_level = ctrl.coarsen_level
        if ctrl.refine_model.double_base:  # if it is double-base, it will need to do one more layer of coarsening
            coarsen_level += 1
        for i in range(coarsen_level):
            match, coarse_graph_size = coarse_method(ctrl, graph)
            coarse_graph = create_coarse_graph(ctrl, graph, match, coarse_graph_size)
            graph = coarse_graph
            if graph.node_num <= ctrl.embed_dim:
                ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
                exit(0)
        print_coarsen_info(ctrl, original_graph)

    timer.printIntervalTime(name='graph coarsening')

    # Base embedding
    if ctrl.refine_model.double_base:
        graph = graph.finer
    embedding = base_embed(ctrl, graph)
    embedding = normalized(embedding, per_feature=False)
    timer.printIntervalTime(name='embedding')

    # Refinement
    if ctrl.coarsen_level > 0:
        np.random.seed(ctrl.seed)
        tf.random.set_seed(ctrl.seed)

        timer3 = Timer(logger=ctrl.logger, ident=1)
        if ctrl.refine_model.double_base:
            coarse_embed = base_embed(ctrl, graph.coarser)
            coarse_embed = normalized(coarse_embed, per_feature=False)
        else:
            coarse_embed = None
        timer3.printIntervalTime(name='double-base embedding')

        # Initialize and train
        tf.config.threading.set_intra_op_parallelism_threads(ctrl.workers)
        ctrl.attr_dim = graph.sens_dim
        ctrl.attr_num = graph.sens_num
        ctrl.attr_range = graph.attr_range
        model = refine_model(ctrl)
        model.train(coarse_graph=graph.coarser, fine_graph=graph, coarse_embed=coarse_embed,
                    fine_embed=embedding)
        timer3.printIntervalTime('training the model')

        # debias_model = Debias(ctrl)
        # debias_model.train(fine_graph=graph, fine_embed=embedding)

        # Apply
        count_lvl = ctrl.coarsen_level
        while graph.finer is not None:  # apply the refinement model.
            embedding = model.predict(coarse_graph=graph, fine_graph=graph.finer, coarse_embed=embedding,
                                      last_level=(count_lvl == 1))
            # embedding = debias_model.predict(embedding)
            graph = graph.finer
            ctrl.logger.info("\t\t\tRefinement at level %d completed." % count_lvl)
            count_lvl -= 1
        timer3.printIntervalTime('refinement')
        embedding = embedding.numpy()

    ctrl.embed_time = timer.restart(title='refinement training and applying', name='main program')

    return embedding


def call_baselines(ctrl, graph, method):
    timer = Timer(logger=ctrl.logger, ident=0)
    embeddings = method(ctrl, graph)
    ctrl.embed_time = timer.restart(title=f'baseline ({method})', name='main program')
    return embeddings