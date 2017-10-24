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
    known_cases = [
        "bit-counter",
        "segment-counter",
        "dense",
        "parity",
        "one-hot-bit",
        "mnist",
        "glass",
        "yeast",
        "wine quality"
    ]

    def __init__(self, file=None):
        with open(file, "r") as f:
            input = json.loads(" ".join(f.readlines()))

        # DATASET:
        self.dataset                    = input["dataset"]["name"]
        self.total_case_fraction        = input["dataset"]["total_case_fraction"]
        self.validation_fraction        = input["dataset"]["validation_fraction"]
        self.test_fraction              = input["dataset"]["test_fraction"]
        self.normalize = input["dataset"]["normalize"] if "normalize" in input["dataset"] else False

        # NETWORK CONFIGURATION:
        self.input_vector_size          = input["network_configuration"]['input_vector_size']
        self.network_dimensions         = input["network_configuration"]['network_dimensions']
        self.validation_interval        = input["network_configuration"]['validation_test_interval']
        self.lower_bound_weight_range   = input["network_configuration"]['weight_range']['lower']
        self.upper_bound_weight_range   = input["network_configuration"]['weight_range']['upper']
        self.hidden_activation          = ActivationFunction(input["network_configuration"]['hidden_activation'])
        self.output_activation          = ActivationFunction(input["network_configuration"]['output_activation'])
        self.cost_function              = CostFunction(input["network_configuration"]['cost_function'])
        self.optimizer_options          = input["network_configuration"]["optimizer"]
        self.optimizer                  = self.select_optimizer(self.optimizer_options["algorithm"])

        # RUN CONFIGURATION
        self.mini_batch_size            = input["run_configuration"]["mini-batch-size"]
        self.epochs                     = input["run_configuration"]["epochs"]
        self.in_top_k_test              = input["run_configuration"]["in_top_k"]

        # VISUALISATION
        self.display_interval           = input["visualisation"]["display_interval"]
        self.map_layers                 = input["visualisation"]["map_layers"]
        self.probe_layers               = input["visualisation"]["probe_layers"]

        classes = input["dataset"]["number_of_classes"] if self.dataset in ["yeast", "glass", "wine quality"] else 0

        if self.validation_interval == "half": self.validation_interval = self.epochs/2

        self.validate()

        # SETTING UP CASEMANAGER:
        self.manager = CaseManager(
            case=self.dataset,
            normalize=self.normalize,
            case_fraction=self.total_case_fraction,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            number_of_classes=classes
        )

    def print_summary(self):
        print("\nSUMMARY OF CONFIGURATION: ")
        print("\tDataset used: " + self.dataset)
        print("\tNetwork dimensions: " + str(self.network_dimensions))
        print("\tOptimizer used: " + str(self.optimizer))
        print("\tSettings: " + str(self.optimizer_options))
        print("\tActivation functions: ")
        print("\t\tHidden layers: " + str(self.hidden_activation))
        print("\t\tOutput layer:  " + str(self.output_activation))
        print("\tCost function: " + str(self.cost_function))

    def validate(self):
        """ Checks the different parameters for errors and bad values. """

        # Case manager configuration:
        if self.dataset not in self.known_cases:
            raise ValueError("Unknown dataset/case given.")
        if self.total_case_fraction > 1:
            raise ValueError("Case fraction cannot be bigger than 1.0")
        if self.validation_fraction > 1:
            raise ValueError("Validation fraction cannot be bigger than 1.0")
        if self.test_fraction > 1:
            raise ValueError("Test fraction cannot be bigger than 1.0")
        if self.validation_fraction + self.test_fraction > 1:
            raise ValueError("The combined value of validation-test fraction is >1. No space for training.")

        if not (0 <= self.hidden_activation.value <= 6):
            raise ValueError("Invalid activation function for hidden layer.")
        if not (0 <= self.output_activation.value <= 6):
            raise ValueError("Invalid activation function for hidden layer.")
        if not (0 <= self.cost_function.value <= 2):
            raise ValueError("Invalid cost function selected.")
        if not (0 <= self.optimizer.value <= 10):
            raise ValueError("Invalid optimizer selected.")

        if self.cost_function == 1 and self.output_activation != 4:
            raise ValueError("Using cross entropy can only be used when output activation function is softmax")

        if self.epochs <= self.validation_interval:
            raise ValueError("Having a validation interval that is equal or bigger that epochs can cause bugs.")

        # if self.softmax_outputs and self.cost_function in [1, 2]:
        #     self.softmax_outputs = False
        # if not self.softmax_outputs and self.cost_function in [1, 2]:
        #     ValueError("Softmax has to be used with cross entropy.")

    def export(self):
        """ Creates a dictionary out of all the values and then dumps to json. """
        with open("configurations/one-hot.json", "w") as f:
            f.write( json.dumps( self.to_dict()) )

    def to_dict(self):
        # TODO: Update this to contain all values.
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

    def select_optimizer(self, param):
        if isinstance(param, str):
            if param.lower() == "adam":
                return Optimizer.ADAM
            if param.lower() == "momentum":
                return Optimizer.MOMENTUM
            if param.lower() in ["sgd", "gradient decent"]:
                return Optimizer.GRADIENT_DECENT
        else:
            return Optimizer(param)