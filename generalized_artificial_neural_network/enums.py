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
    SIGMOID = 0
    RECTIFIED_LINEAR = 1
    EXPONENTIAL_LINEAR = 2
    HYPERBOLIC_TANGENT = 3
    SOFTPLUS = 4
    SOFTSIGN = 5
    DROPOUT = 6

class CostFunction(Enum):
    MEAN_SQUARED_ERROR = 0
    CROSS_ENTROPY = 1

class Optimizer(Enum):
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
    GRADIENT_DECENT = 0
    ADADELTA = 1
    ADAGRAD = 2
    ADAGRADDA = 3
    MOMENTUM = 4
    ADAM = 5
    FTRL = 6
    PROXIMAL_GRADIENT_DECENT = 7
    PROXIMAL_ADAGRAD = 8
    RMS_PROP = 9