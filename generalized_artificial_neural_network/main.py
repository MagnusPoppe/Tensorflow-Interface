import os

import time

from generalized_artificial_neural_network.trainer import Trainer


def mark_the_selected_variables_for_monitoring(configuration, network):
    """ Add a grabbed variable, or a variable that is taken
        out of the graph to be displayed.
        (to be displayed in its own matplotlib window).
     """
    for layer in configuration.map_layers:
        network.net.add_grabvar(layer["placement"], layer["component"])

def mark_the_selected_modules_for_probing(configuration, coach):
    """ Grabs a variable for use with tensorboard """
    for layer in configuration.probe_layers:
        coach.ann.generate_probe(
            layer["placement"],
            layer["component"],
            (layer["points_of_interest"][0], layer["points_of_interest"][1]))



start_time = time.time()
file = "bit-counter.json"  # "glass.json" #"one-hot.json"
configuration_file = os.path.join("configurations", file)

coach = Trainer(configuration_file, display_graph=False)
mark_the_selected_modules_for_probing(configuration=coach.config, coach=coach)

coach.train(epochs=coach.config.epochs)
print("\nTRAINING COMPLETE!")
print("\tERROR AFTER TRAINING: " + str(coach.error_history[-1][1]))

# Running tests:
training_score = coach.test(coach.config.manager.get_training_cases(), in_top_k=True)
validation_score = coach.test(coach.config.manager.get_validation_cases(), in_top_k=True)
testing_score = coach.test(coach.config.manager.get_testing_cases(), in_top_k=True)

print("\nPERFORMING TESTS:")
print("\tSCORE ON TRAINING CASES:   " + str(training_score)   + " %")
print("\tSCORE ON VALIDATION CASES: " + str(validation_score) + " %")
print("\tSCORE ON TESTING CASES:    " + str(testing_score)    + " %")

print("\nTime used for this run: " + str(time.time()-start_time))
run = True
waited = 0
while run:
    x = input(">>> ")

    if x in ["q", "quit", "exit"]: run = False
    elif x in ["tb", "tensorboard"]: coach.run_tensorboard()
    elif x == "run more": pass
    elif x in "run tests": coach.test(coach.config.manager.get_testing_cases())