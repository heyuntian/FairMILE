"""
Layers in refinement models.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers
spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul

def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse and dense).
    From GitHub @jiongqian/MILE
    https://github.com/jiongqian/MILE/blob/master/refine_model.py
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class scale_weights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        return w - tf.reduce_mean(w, axis=0)


class GCNConv(tf.keras.layers.Layer):
    """
    From Github @cshjin/GCN-TF2.0
    https://github.com/cshjin/GCN-TF2.0/blob/15232a7da73dbca0591a0f8551d7b0fc4495f3de/models/layers.py
    """
    def __init__(self,
                 output_dim,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        # gsize = input_shape[0][0]  # graph size
        input_dim = input_shape[1][1]  # feature dim

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(input_dim, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.output_dim,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GCNConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ GCN has two inputs : [An, X]
        :param **kwargs:
        """
        self.An = inputs[0]
        self.X = inputs[1]

        output = dot(self.An, dot(self.X, self.weight, sparse=isinstance(self.X, tf.SparseTensor)), sparse=True)
        # if isinstance(self.X, tf.SparseTensor):
        #     h = spdot(self.X, self.weight)
        # else:
        #     h = dot(self.X, self.weight)
        # output = spdot(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output