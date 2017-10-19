import json
from generalized_artificial_neural_network.case_manager import CaseManager
from generalized_artificial_neural_network.enums import ActivationFunction, CostFunction, Optimizer

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
        with open(file, "r") as f:
            input = json.loads(" ".join(f.readlines()))

        # DATASET:
        self.dataset                    = input["dataset"]["name"]
        self.total_case_fraction        = input["dataset"]["total_case_fraction"]
        self.steps_per_minibatch        = input["dataset"]["steps_per_minibatch"]
        self.validation_fraction        = input["dataset"]["validation_fraction"]
        self.test_fraction              = input["dataset"]["test_fraction"]

        # NETWORK CONFIGURATION:
        self.network_dimensions         = input["network_configuration"]['network_dimensions']
        self.learning_rate              = input["network_configuration"]['learning_rate']
        self.validation_interval        = input["network_configuration"]['validation_test_interval']
        self.softmax                    = input["network_configuration"]['use_softmax']
        self.lower_bound_weight_range   = input["network_configuration"]['weight_range']['lower']
        self.upper_bound_weight_range   = input["network_configuration"]['weight_range']['upper']
        self.hidden_activation          = ActivationFunction(input["network_configuration"]['hidden_activation'])
        self.output_activation          = ActivationFunction(input["network_configuration"]['output_activation'])
        self.cost_function              = CostFunction(input["network_configuration"]['cost_function'])
        self.optimizer                  = Optimizer(input["network_configuration"]["optimizer"])

        # RUN CONFIGURATION
        self.mini_batch_size            = input["run_configuration"]["mini-batch-size"]
        self.map_batch_size             = input["run_configuration"]["map-batch-size"] # todo: use this
        self.epochs                     = input["run_configuration"]["epochs"]

        # VISUALISATION
        self.display_interval           = input["visualisation"]["display_interval"] # todo: is this in use?
        self.map_layers                 = input["visualisation"]["map_layers"] # todo: is this in use?
        self.map_dendrograms            = input["visualisation"]["map_dendrograms"] # todo: is this in use?
        self.display_weights            = input["visualisation"]["display_weights"] # todo: is this in use?
        self.display_biases             = input["visualisation"]["display_biases"] # todo: is this in use?

        # SETTING CASEMANAGER:
        self.manager = CaseManager(
            self.dataset,
            self.mini_batch_size,
            self.total_case_fraction,
            self.validation_fraction,
            self.test_fraction
        )

    def export(self):
        """ Creates a dictionary out of all the values and then dumps to json. """
        with open("configurations/one-hot.json", "w") as f:
            f.write( json.dumps( self.to_dict()) )

    def to_dict(self):
        return {
            "dataset": {
                "name": self.dataset,
                "total_case_fraction": self.total_case_fraction,
                "steps_per_minibatch": self.steps_per_minibatch,
                "validation_fraction": self.validation_fraction,
                "test_fraction": self.test_fraction
            },
            "network_configuration": {
                'network_dimensions': self.network_dimensions,
                'learning_rate': self.learning_rate,
                'weight_range': {
                    'lower': self.lower_bound_weight_range,
                    'upper': self.upper_bound_weight_range
                },
                'validation_test_interval': self.validation_interval,
                'use_softmax': self.softmax,
                'hidden_activation': self.hidden_activation.value,
                'output_activation': self.output_activation.value,
                'cost_function': self.cost_function.value,
                'optimizer': self.optimizer
            },
            "run_configuration": {
                "mini-batch-size": self.mini_batch_size,
                "map-batch-size": self.map_batch_size,
                "epochs": self.epochs
            },
            "visualisation": {
                'display_interval': self.display_interval,
                "map_layers": self.map_layers,
                "map_dendrograms": self.map_dendrograms,
                "display_weights": self.display_weights,
                "display_biases": self.display_biases
            }
        }