from generalized_artificial_neural_network.layer import Layer
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration

import tensorflow as tf
import numpy as np

class NeuralNetwork:

    def __init__(self, configuration:NetworkConfiguration):
        self.hidden_layers = []
        self.config = configuration
        self._build()
        self._learning_setup()

    def _build(self):
        """
        Builds up the network, from input vector to output vector
        with all hidden layers inbetween.
        """

        tf.reset_default_graph()
        output_layer = len(self.config.network_dimensions)-1

        # Creating the input layer. This is a vector that is feed through the
        # feed_dict parameter of the session.run() method. This is the inputs.
        # The vector therefore needs to be the exact dimensions of the input
        # vector:
        self.input = tf.placeholder(
            tf.float64,
            shape=(None, self.config.input_vector_size),
            name="input"
        )

        # The layer creation loop creates the network layer for layer, connecting
        # the previous vector to a weight matrix and a bias vector. The input
        # layer is then multiplied to create the next vector.
        previous_vector = self.input
        previous_vector_size = self.config.input_vector_size

        # Creating the hidden layers:
        for i, next_vector_size in enumerate(self.config.network_dimensions):
            if i == output_layer:
                activation = self.config.hidden_activation
            else:
                activation = self.config.output_activation

            # Creating the actual layer:
            layer = Layer(self, i, previous_vector, previous_vector_size, next_vector_size, activation,
                          self.config.lower_bound_weight_range, self.config.upper_bound_weight_range)

            # Setting the variables for the next layer to use:
            previous_vector = layer.output_vector
            previous_vector_size = layer.out_neurons

            # Storing the layer:
            self.hidden_layers += [ layer ]

        # The last layer created contains the output vector.
        self.output = self.hidden_layers[-1].output_vector

    def _learning_setup(self):
        # The target we want to compare to needs to be the same dimensions as the output:
        self.target = tf.placeholder(tf.float64, shape=(None, self.hidden_layers[-1].out_neurons), name="Target")

        # The error function here is the "mean squared error".
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name="Mean-squared-error")

        # This is the function used to optimize the weights of the layers in the network.
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        self.trainer   = self.optimizer.minimize(self.error, name="Backpropogation")


