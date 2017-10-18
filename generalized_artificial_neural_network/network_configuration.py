import json

from generalized_artificial_neural_network.enums import ActivationFunction


class NetworkConfiguration:

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

    def _import(self):
        with open("configurations/exported.json", "r") as f:
            input = json.loads(" ".join(f.readlines()))

        self.network_dimensions = input['network_dimensions']
        self.learning_rate = input['learning_rate' ]
        self.display_interval = input['display_interval']
        self.mbs = input['mini-batch-size']
        self.validation_interval = input['validation_test_interval']
        self.softmax = input['use_softmax']
        self.hidden_layer_activation_function = input['hidden_layer_activation_function']
        self.output_layer_activaiton_function = input['output_layer_activaiton_function']
    def _export(self):
        output = {
            'network_dimensions': self.network_dimensions,
            'learning_rate': self.learning_rate,
            'display_interval': self.display_interval,
            'mini-batch-size' : self.mbs,
            'validation_test_interval': self.validation_interval,
            'use_softmax': self.softmax,
            'hidden_layer_activation_function': self.hidden_layer_activation_function.value,
            'output_layer_activaiton_function': self.output_layer_activaiton_function.value
        }
        with open("configurations/exported.json", "w") as f:
            f.write(json.dumps(output))