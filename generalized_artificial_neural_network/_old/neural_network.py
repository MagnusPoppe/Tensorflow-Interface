import matplotlib.pyplot as PLT
import tensorflow as tf

from generalized_artificial_neural_network._old.network_layer import NetworkLayer
from generalized_artificial_neural_network.enums import CostFunction, Optimizer
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration


class NeuralNetwork():

    def __init__(self, configuration: NetworkConfiguration):

        # Specifications of the network:
        self.layer_dimensions = configuration.network_dimensions   # Sizes of each layer of neurons
        self.minibatch_size = configuration.mini_batch_size
        self.lower_bound_weight_value = configuration.lower_bound_weight_range
        self.upper_bound_weight_value = configuration.upper_bound_weight_range
        self.modules = []

        # Cost function:
        self.cost_function = configuration.cost_function

        # Activation functions:
        self.hidden_layer_activation_function = configuration.hidden_activation
        self.output_layer_activation_function = configuration.output_activation
        # self.softmax_outputs = configuration.softmax_outputs

        # Learning
        self.learning_rate = configuration.learning_rate
        self.momentum = configuration.momentum
        self.optimizer_function = configuration.optimizer

        # Case management:
        self.caseman = configuration.manager

        # Monitored values:
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.

        # Visuals
        self.show_interval = configuration.display_interval # Frequency of showing grabbed variables
        self.grabvar_figures = [] # One matplotlib figure for each grabvar

        # Other information:
        # Enables coherent data-storage during extra training runs (see runmore).
        self.global_training_step = 0

        # Building the network:
        self.build()

    def gen_probe(self, module_index, type, spec):
        """ Probed variables are to be displayed in the Tensorboard. """
        self.modules[module_index].generate_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module):
        self.modules.append(module)

    def build(self):
        # Resetting the graph:
        tf.reset_default_graph()  # This is essential for doing multiple runs!!

        # Defining the first layer, a.k.a Input layer:
        self.input = tf.placeholder(tf.float64, shape=(None, self.layer_dimensions[0]), name='Input')
        in_layer = self.input
        in_layer_size = self.layer_dimensions[0]

        # Build all of the modules
        for placement, out_layer_size in enumerate(self.layer_dimensions[1:]):
            # Choosing the correct activation function:
            if placement +1 != len(self.layer_dimensions) -1:
                activation = self.hidden_layer_activation_function
            else:
                activation = self.output_layer_activation_function

            # Creating the hidden layer:
            layer = NetworkLayer(self,
                 placement,
                 in_layer,
                 in_layer_size,
                 out_layer_size,
                 self.lower_bound_weight_value,
                 self.upper_bound_weight_value,
                 activation
             )

            # Defining the in-layer for the next level in the neural network:
            in_layer = layer.out_layer
            in_layer_size = layer.out_neurons

        # After the network has been built, we store the last layer as output layer:
        self.output = layer.out_layer

        # Adding softmax to outputs:
        # if self.softmax_outputs: self.output = tf.nn.softmax(self.output)

        # Adding a target value for the net. This is a placeholder. The value comes from the datasets.
        self.target = tf.placeholder(tf.float64, shape=(None,layer.out_neurons),name='Target')
        self.configure_learning()

    def configure_learning(self):
        self.error = self.select_cost_function()

        # Predictor. Denne hjelper for å hente ut verdier under testing ved å hente verdier fra output:
        self.predictor = self.output

        # Defining the training operator
        optimizer = self.select_optimizer()
        self.trainer = optimizer.minimize(self.error,name='Backpropagation')

    def select_optimizer(self) -> tf.train:
        if self.optimizer_function == Optimizer.GRADIENT_DECENT:
            return tf.train.GradientDescentOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.ADADELTA:
            return tf.train.AdadeltaOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.ADAGRAD:
            return tf.train.AdagradOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.ADAGRADDA:
            return tf.train.AdagradDAOptimizer(self.learning_rate, self.global_training_step)
        if self.optimizer_function == Optimizer.MOMENTUM:
            return tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        if self.optimizer_function == Optimizer.NAG:
            return tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
        if self.optimizer_function == Optimizer.ADAM:
            return tf.train.AdamOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.FTRL:
            return tf.train.FtrlOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.PROXIMAL_GRADIENT_DECENT:
            return tf.train.ProximalGradientDescentOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.PROXIMAL_ADAGRAD:
            return tf.train.ProximalAdagradOptimizer(self.learning_rate)
        if self.optimizer_function == Optimizer.RMS_PROP:
            return tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)

    def select_cost_function(self):
        # Cost Function used to create calculate loss/error:
        if self.cost_function == CostFunction.MEAN_SQUARED_ERROR:
            return tf.reduce_mean(
                tf.square(self.target - self.output),
                name='MEAN-SQUARED-ERROR'
            )

        if self.cost_function == CostFunction.CROSS_ENTROPY_TENSORFLOW_DEFAULT:
            # Tensorflow implementation of cross-entropy
            return tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.target),
                name="CROSS-ENTROPY-ERROR"
            )

        if self.cost_function == CostFunction.CROSS_ENTROPY_FORMULA:
            # My implementation of cross-entropy (not as stable as tensorflow's algorithm)
            self.output = tf.nn.softmax(self.output)
            return - tf.reduce_mean(
                tf.reduce_sum(self.target * tf.log(self.output), [1]) ,
                name='CROSS-ENTROPY-ERROR'
            )