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

    EXTREMLY GOOD EXAMPLE OF OPTIMIZERS:
    https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow

    ADAPTIVE GRADIENT
    AdaGrad or adaptive gradient allows the learning rate to adapt based on parameters. It performs larger updates
    for infrequent parameters and smaller updates for frequent one. Because of this it is well suited for sparse
    data (NLP or image recognition). Another advantage is that it basically eliminates the need to tune the learning
    rate. Each parameter has its own learning rate and due to the peculiarities of the algorithm the learning rate is
    monotonically decreasing. This causes the biggest problem: at some point of time the learning rate is so small
    that the system stops learning

    ADAPTIVE DELTA
    AdaDelta resolves the problem of monotonically decreasing learning rate in AdaGrad. In AdaGrad the learning rate
    was calculated approximately as one divided by the sum of square roots. At each stage you add another square root
    to the sum, which causes denominator to constantly decrease. In AdaDelta instead of summing all past square roots
    it uses sliding window which allows the sum to decrease. RMSprop is very similar to AdaDelta

    MOMENTUM:
    momentum helps SGD to navigate along the relevant directions and softens the oscillations in the irrelevant.
    It simply adds a fraction of the direction of the previous step to a current step. This achieves amplification
    of speed in the correct direction and softens oscillation in wrong directions. This fraction is usually in the
    (0, 1) range. It also makes sense to use adaptive momentum. In the beginning of learning a big momentum will
    only hinder your progress, so it makes sense to use something like 0.01 and once all the high gradients
    disappeared you can use a bigger momentum. There is one problem with momentum: when we are very close to the
    goal, our momentum in most of the cases is very high and it does not know that it should slow down. This can
    cause it to miss or oscillate around the minima

    ADAPTIVE MOMENTUM
    Adam or adaptive momentum is an algorithm similar to AdaDelta. But in addition to storing learning rates for
    each of the parameters it also stores momentum changes for each of them separately

    NESTROV GRADIENT ACCELERATOR
    nesterov accelerated gradient overcomes this problem by starting to slow down early. In momentum we first
    compute gradient and then make a jump in that direction amplified by whatever momentum we had previously.
    NAG does the same thing but in another order: at first we make a big jump based on our stored information,
    and then we calculate the gradient and make a small correction. This seemingly irrelevant change gives
    significant practical speedups.
    """
    GRADIENT_DECENT = 0             # tf.train.GradientDescentOptimizer
    ADAGRADDA = 1                   # tf.train.AdagradDAOptimizer
    FTRL = 2                        # tf.train.FtrlOptimizer
    PROXIMAL_GRADIENT_DECENT = 3    # tf.train.ProximalGradientDescentOptimizer
    PROXIMAL_ADAGRAD = 4            # tf.train.ProximalAdagradOptimizer
    RMS_PROP = 5                    # tf.train.RMSPropOptimizer
    ADAGRAD = 6                     # tf.train.AdagradOptimizer
    ADADELTA = 7                    # tf.train.AdadeltaOptimizer
    MOMENTUM = 8                    # tf.train.MomentumOptimizer
    ADAM = 9                        # tf.train.AdamOptimizer
    NAG = 10                        # tf.train.MomentumOptimizer (use_nestrov=True)