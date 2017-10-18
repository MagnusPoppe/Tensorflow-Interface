import tensorflow as tf
import numpy as np

from generalized_artificial_neural_network.enums import ActivationFunction


class NetworkLayer():
    """
    A layer inside a neural network. This consists of:
    1. The in-layer
    2. The weights between in-layer and out-layer
    3. The bias that is added to the out-layer results
    4. The layer, aka. an activation function that results in a output.
    """

    def __init__(self, network, placement_in_network, in_layer, in_neurons, out_neurons, activation_func=ActivationFunction.RECTIFIED_LINEAR):
        """
        Sets the needed values for the layer. Also builds it up.
        :param network: The neural network it belongs to a.k.a parent
        :param placement_in_network: index of layer inside the network.
        :param in_layer: Either the gann's input variable or the upstream module's output
        :param in_neurons: Number of neurons feeding into this module
        :param out_neurons: Number of neurons in this module
        :param activation_func: Activation function used on the output layer
        """
        # The neural network it belongs to a.k.a parent
        self.network = network

        # Definition of the input layer:
        self.in_neurons  = in_neurons # Number of neurons feeding into this module
        self.in_layer    = in_layer   # Either the gann's input variable or the upstream module's output

        # Size of this layer:
        self.out_neurons=out_neurons # Number of neurons in this module

        # Activation function used on the output layer:
        if activation_func == ActivationFunction.RECTIFIED_LINEAR:
            self.activation_function = tf.nn.relu
        elif activation_func == ActivationFunction.SIGMOID:
            self.activation_function = tf.sigmoid
        elif activation_func == ActivationFunction.EXPONENTIAL_LINEAR:
            self.activation_function = tf.nn.elu
        elif activation_func == ActivationFunction.SOFTPLUS:
            self.activation_function = tf.nn.softplus
        elif activation_func == ActivationFunction.SOFTSIGN:
            self.activation_function = tf.nn.softsign
        elif activation_func == ActivationFunction.HYPERBOLIC_TANGENT:
            self.activation_function = tf.tanh

        # Other information for visualization (e.g. TensorBoard):
        self.placement_in_network = placement_in_network        # index.
        self.name = "Module-"+str(self.placement_in_network)    # name for visuals.

        # Building network:
        self.build()

    def build(self):
        """
        Builds up the network.

        THE WEIGHTS:
        Since the in-layer is known, we build the weight matrix in the dimensions of
        the input layer size x the output layer size. The value of the weights resides
        between -1 and 1. They are of course trainable by default. This has to be
        turned of after training is done.

        THE BIAS:
        The Bias defaults in the same range as the weights. Same dimensions as the
        out-layer.

        THE OUT LAYER
        the out_layer is created by using a given activation function and
        a matrix multiplication of the in-layer and the weight-matrix. The
        bias is added after the multiplication.
        """

        # Creates the weights and biases:
        self.weights = tf.Variable(
            np.random.uniform(-.1, .1, size = (self.in_neurons,self.out_neurons)),
            name            =  self.name+'-wgt',
            trainable       = True
        )
        self.biases  = tf.Variable(
            np.random.uniform(-.1, .1, size = self.out_neurons),
            name            = self.name+'-bias',
            trainable       = True
        )

        # Creates the output layer.
        self.out_layer = self.activation_function(
            tf.matmul(self.in_layer, self.weights) + self.biases,
            name=self.name + '-out'
        )
        self.network.add_module(self)

    def getvar(self, type:tuple):
        """
        Gets a variable inside the network for visualizing and debugging.
        :param type: (in,out,wgt,bias)
        :return: dict.
        """
        return {'in': self.in_layer, 'out': self.out_layer, 'wgt': self.weights, 'bias': self.biases}[type]

    def generate_probe(self, type:tuple, spec:list):
        """
        :param type: (in, out, wgt, bias)
        :param spec: a list, can contain one or more of (avg,max,min,hist);
        """
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)