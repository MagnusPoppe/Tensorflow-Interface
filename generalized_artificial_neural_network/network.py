from generalized_artificial_neural_network.enums import Optimizer
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
            if i != output_layer: activation = self.config.hidden_activation
            else:                 activation = self.config.output_activation

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

        # Predictor is used as the operator when testing.
        self.predictor = self.output

        # The error function here is the "mean squared error".
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name="Mean-squared-error")

        # This is the function used to optimize the weights of the layers in the network.
        if self.config.optimizer == Optimizer.GRADIENT_DECENT:
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.config.optimizer_options["learning_rate"],
                use_locking=self.config.optimizer_options["locking"]
            )
        elif self.config.optimizer == Optimizer.MOMENTUM:
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.config.optimizer_options["learning_rate"],
                momentum=self.config.optimizer_options["momentum"],
                use_nesterov=self.config.optimizer_options["use_nestrov"],
                use_locking=self.config.optimizer_options["locking"]
            )
        elif self.config.optimizer == Optimizer.ADAM:
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.optimizer_options["learning_rate"],
                beta1=self.config.optimizer_options["Beta1 (exponential decay rate for the 1st moment estimates)"],
                beta2=self.config.optimizer_options["Beta2 (exponential decay rate for the 2st moment estimates)"],
                epsilon=self.config.optimizer_options["epsilon"],
                use_locking=self.config.optimizer_options["locking"]
            )
        else: raise Exception("Invalid optimizer. Use a different optimizer.")
        self.trainer   = self.optimizer.minimize(self.error, name="Backpropogation")

    def generate_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """
        self.hidden_layers[module_index].generate_probe(type,spec)