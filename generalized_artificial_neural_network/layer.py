import tensorflow as tf
import numpy as np

class Layer():
    """
    Builds up the Layer.

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

    def __init__(self, network, index, input_layer, in_neurons, out_neurons, activation, lower, upper):
        # Saving initial variables:
        self.network             = network
        self.input_vector        = input_layer
        self.in_neurons          = in_neurons
        self.out_neurons         = out_neurons
        self.activation_function = activation
        self.index               = index

        # Name of module to be used with tensorboard and other visuals.
        self.name = "Hidden_layer_"+str(index)

        # Creates the weights and biases:
        self.weight_matrix = tf.Variable(
            np.random.uniform(lower, upper, size=(self.in_neurons, self.out_neurons)),
            name=self.name + '_weight',
            trainable=True
        )
        self.bias_vector = tf.Variable(
            np.random.uniform(lower, upper, size=self.out_neurons),
            name=self.name + '_bias',
            trainable=True
        )

        # TODO: Implement more activation functions:
        activation_function = tf.nn.relu

        # Creating the output layer
        self.output_vector = activation_function(
            tf.matmul(self.input_vector, self.weight_matrix) + self.bias_vector,
            name=self.name + '_output'
        )