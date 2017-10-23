import os
import sys

from generalized_artificial_neural_network._old.network_controller import NetworkController
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration


def mark_the_selected_variables_for_grabbing(configuration):
    """ Add a grabbed variable, or a variable that is taken
        out of the graph to be displayed.
        (to be displayed in its own matplotlib window).
     """
    for layer in configuration.map_layers:
        ann.net.add_grabvar(layer["placement"], layer["component"])



def mark_the_selected_modules_for_probing(configuration):
    """ Grabs a variable for use with tensorboard """
    for layer in configuration.probe_layers:
        ann.net.gen_probe(
            layer["placement"],
            layer["component"],
            (layer["points_of_interest"][0], layer["points_of_interest"][1]))

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

    mark_the_selected_variables_for_grabbing(c)
    mark_the_selected_modules_for_probing(c)

    ann.run(c.epochs, continued=False)
    # ann.save_session_params("netsaver/saved_dense")
    print("\n\n RUNNING MORE... \n")
    ann.runmore(c.epochs*2)
    # # return ann
