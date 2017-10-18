import json

from generalized_artificial_neural_network.enums import ActivationFunction, CostFunction


class NetworkConfiguration:
    """ TIPS HENTET FRA STACK OVERFLOW:
    https://stackoverflow.com/questions/34229140/choosing-from-different-cost-function-and-activation-function-of-a-neural-networ
    What to use. Now to the last question, how does one choose which activation and cost
    functions to use. These advices will work for majority of cases:

    1.  If you do classification, use softmax for the last layer's nonlinearity and cross
        entropy as a cost function.

    2.  If you do regression, use sigmoid or tanh for the last layer's nonlinearity and
        squared error as a cost function.

    3.  Use ReLU as a nonlienearity between layers.

    4.  Use better optimizers(AdamOptimizer, AdagradOptimizer) instead of
        GradientDescentOptimizer, or use momentum for faster convergence,
    """

    def __init__(self, file=None):
        if file:
            self._import()
        else:
            self.network_dimensions = None
            self.manager = None
            self.learning_rate = 0.1
            self.display_interval = None
            self.mbs = 10
            self.validation_interval = None
            self.softmax = False
            self.hidden_layer_activation_function = ActivationFunction.RECTIFIED_LINEAR
            self.output_layer_activaiton_function = ActivationFunction.SIGMOID
            self.cost_function = CostFunction.MEAN_SQUARED_ERROR

    def _import(self):
        with open("configurations/exported.json", "r") as f:
            input = json.loads(" ".join(f.readlines()))

        self.network_dimensions = input['network_dimensions']
        self.learning_rate = input['learning_rate' ]
        self.display_interval = input['display_interval']
        self.mbs = input['mini-batch-size']
        self.validation_interval = input['validation_test_interval']
        self.softmax = input['use_softmax']
        self.hidden_layer_activation_function = ActivationFunction(input['hidden_layer_activation_function'])
        self.output_layer_activaiton_function = ActivationFunction(input['output_layer_activaiton_function'])
        self.cost_function = CostFunction(input['cost_function'])

    def _export(self):
        output = {
            'network_dimensions': self.network_dimensions,
            'learning_rate': self.learning_rate,
            'display_interval': self.display_interval,
            'mini-batch-size' : self.mbs,
            'validation_test_interval': self.validation_interval,
            'use_softmax': self.softmax,
            'hidden_layer_activation_function': self.hidden_layer_activation_function.value,
            'output_layer_activaiton_function': self.output_layer_activaiton_function.value,
            'cost_function': self.cost_function.value
        }
        with open("configurations/exported.json", "w") as f:
            f.write(json.dumps(output))