"""
Config for coarsening, embedding, and refinement
"""

import numpy as np
import tensorflow as tf


class Config:
    def __init__(self):
        # Basics
        self.dataset = None
        self.logger = None
        self.embed_dim = 128
        self.coarsen_level = 0

        # Coarsening
        self.max_node_wgt = 100
        self.wgt_merge = 1

        # Embedding
        self.basic_embed = "deepwalk"
        self.workers = 4

        # Refinement
        self.refine_type = "gcn"
        self.refine_model = RefineModelSetting()

        # Baselines
        self.fairwalk = FairWalkSetting()


class RefineModelSetting:
    def __init__(self):
        self.double_base = False
        self.learning_rate = 0.001
        self.epoch = 200
        self.early_stopping = 50  # Tolerance for early stopping (# of epochs).
        self.wgt_decay = 5e-4
        self.regularized = True
        self.hidden_layer_num = 2
        self.act_func = tf.keras.activations.tanh   # tf.tanh
        self.tf_optimizer = tf.keras.optimizers.Adam
        self.lda = 0.05  # self-loop weight lambda
        self.lambda_fl = 1
        self.negative = 50
        self.fair_threshold = 0.5


class FairWalkSetting:
    def __init__(self):
        self.attr_id = 0