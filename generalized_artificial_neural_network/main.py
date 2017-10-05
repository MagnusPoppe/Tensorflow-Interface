import downing_code.tflowtools as TFT

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
from generalized_artificial_neural_network.case_manager import CaseManager
from generalized_artificial_neural_network.network_controller import NetworkController


def autoex(epochs=300,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    cman = CaseManager(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = NetworkController(
        network_dimensions=[size,nbits,size],
        manager=cman,
        learning_rate=lrate,
        display_interval=showint,
        mbs=mbs,
        validation_interval=vint,
        softmax=sm)

    ann.net.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.net.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.net.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    ann.runmore(epochs*2)
    return ann
