from enum import Enum

class ActivationFunction(Enum):
    """
    tf.nn.relu
    tf.nn.relu6
    tf.nn.crelu
    tf.nn.elu
    tf.nn.softplus
    tf.nn.softsign
    tf.nn.dropout
    tf.nn.bias_add
    tf.sigmoid
    tf.tanh
    """
    SIGMOID = 2
    RECTIFIED_LINEAR = 1
    EXPONENTIAL_LINEAR = 2
    HYPERBOLIC_TANGENT = 3
    SOFTPLUS = 4
    SOFTSIGN = 5
    DROPOUT = 6

class Optimiser(Enum):
    """
    tf.train.Optimizer
    tf.train.GradientDescentOptimizer
    tf.train.AdadeltaOptimizer
    tf.train.AdagradOptimizer
    tf.train.AdagradDAOptimizer
    tf.train.MomentumOptimizer
    tf.train.AdamOptimizer
    tf.train.FtrlOptimizer
    tf.train.ProximalGradientDescentOptimizer
    tf.train.ProximalAdagradOptimizer
    tf.train.RMSPropOptimizer
    """
    OPTIMISER = 0
    GRADIENT_DECENT = 1
    ADADELTA = 2
    ADAGRAD = 3
    ADAGRADDA = 4
    MOMENTUM = 5
    ADAM = 6
    FTRL = 7
    PROXIMAL_GRADIENT_DECENT = 8
    PROXIMAL_ADAGRAD = 9
    RMS_PROP = 10