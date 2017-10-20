import os
import sys

import downing_code.tflowtools as TFT

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
from generalized_artificial_neural_network.case_manager import CaseManager
from generalized_artificial_neural_network.enums import ActivationFunction
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration
from generalized_artificial_neural_network.network_controller import NetworkController



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("PARAMETER ERROR! \n"
              "File is needed as input for this application. Only one argument is accepted. "
              "The file should be a configfile. \n\nAll preconfigured files resides in:\n "
              ".../module 3/generalized_artificial_neural_network/configurations/\n"
          )
        exit(0)

    file = sys.argv[1]

    c = NetworkConfiguration(os.path.join("configurations", file))
    ann = NetworkController(c)

    # ann.net.gen_probe(0, 'wgt', ('hist', 'avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    # ann.net.gen_probe(1, 'out', ('avg', 'max'))  # Plot average and max value of module 1's output vector
    # ann.net.add_grabvar(0, 'wgt')  # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(c.epochs)
    print("\n\n RUNNING MORE... \n")
    ann.runmore(c.epochs)
    # return ann
