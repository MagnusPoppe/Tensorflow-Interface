import downing_code.tflowtools as TFT

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
from generalized_artificial_neural_network.case_manager import CaseManager
from generalized_artificial_neural_network.enums import ActivationFunction
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration
from generalized_artificial_neural_network.network_controller import NetworkController


def autoex(epochs=10000,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False):

    size = 2**nbits

    c = NetworkConfiguration("configurations/exported.json")

    # c.network_dimensions = [size, nbits, size]
    # c.mbs = mbs if mbs else size
    # c.learning_rate = lrate
    # c.display_interval = showint
    # c.validation_interval = vint
    # c.softmax = sm
    # c.hidden_layer_activation_function = ActivationFunction.SIGMOID
    # c.output_layer_activaiton_function = ActivationFunction.RECTIFIED_LINEAR
    # c._export()

    c.export()
    ann = NetworkController( c )


    ann.net.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.net.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.net.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    ann.runmore(epochs*2)
    return ann

if __name__ == '__main__':
    autoex()