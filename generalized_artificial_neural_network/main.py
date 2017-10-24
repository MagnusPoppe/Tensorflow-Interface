import os

import time

from generalized_artificial_neural_network.trainer import Trainer


def mark_the_selected_modules_for_monitoring(configuration, coach):
    """ Add a grabbed variable, or a variable that is taken
        out of the graph to be displayed.
        (to be displayed in its own matplotlib window).
     """
    for layer in configuration.map_layers:
        coach.monitor_module(layer["placement"], layer["component"])

def mark_the_selected_modules_for_probing(configuration, coach):
    """ Grabs a variable for use with tensorboard """
    for layer in configuration.probe_layers:
        coach.ann.generate_probe(
            layer["placement"],
            layer["component"],
            (layer["points_of_interest"][0], layer["points_of_interest"][1]))



start_time = time.time()
file = "winequality.json" # "bit-counter.json"  # "glass.json" #"one-hot.json"
configuration_file = os.path.join("configurations", file)

coach = Trainer(configuration_file, display_graph=True, hinton_plot=False)
mark_the_selected_modules_for_probing(configuration=coach.config, coach=coach)
mark_the_selected_modules_for_monitoring(configuration=coach.config, coach=coach)

coach.train(epochs=coach.config.epochs)
print("\nTRAINING COMPLETE!")
print("\tERROR AFTER TRAINING: " + str(coach.error_history[-1][1]))


def run_tests(print_stats=True):
    # Perform testing:
    training_score = coach.test(coach.config.manager.get_training_cases(), in_top_k=True)
    validation_score = coach.test(coach.config.manager.get_validation_cases(), in_top_k=True)
    testing_score = coach.test(coach.config.manager.get_testing_cases(), in_top_k=True)

    # Print stats:
    if print_stats:
        print("\nPERFORMING TESTS:")
        print("\tTRAINING CASES:   " + str(training_score) + " % CORRECT")
        print("\tVALIDATION CASES: " + str(validation_score) + " % CORRECT")
        print("\tTESTING CASES:    " + str(testing_score) + " % CORRECT")
    return training_score, validation_score, testing_score

# Running tests:
training_score, validation_score, testing_score = run_tests()

print("\nTime used for this run: " + str(time.time()-start_time))
run = True
waited = 0
while run:
    x = input(">>> ")

    if x in ["q", "quit", "exit"]: run = False
    elif x in ["tb", "tensorboard"]: coach.run_tensorboard()
    elif x in ["display hinton history", "dhh"]: coach.display_hinton_graph_from_training_history()
    elif x in ["mapping"]: coach.mapping(coach.config.manager.testing_cases)
    elif x in ["info","settings", "i"]: coach.config.print_summary()
    elif x == "run more": pass
    elif x in "run tests": coach.test(coach.config.manager.get_testing_cases())
    else: print("Unknown command.")