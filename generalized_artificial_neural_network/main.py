import os

import time

import sys

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

def input_number(text:str) -> int:
    epochs = input("\t"+text)
    while not epochs.isdigit():
        print("\"" + epochs + "\" is not a number. Write a number!")
        epochs = input("\t"+text)
    return int(epochs)

def input_json_file(text:str="file: configurations/") -> str:
    print(os.listdir("configurations/"))
    file = input("\tfile: configurations/")
    while file[-5:] != ".json":
        print("\"" + file + "\" is no a json file!")
        epochs = input("\t"+text)
    return file

def create_artificial_neural_network_from_config_file(file) -> Trainer:
    configuration_file = os.path.join("configurations", file)
    coach = Trainer(configuration_file)
    mark_the_selected_modules_for_probing(configuration=coach.config, coach=coach)
    mark_the_selected_modules_for_monitoring(configuration=coach.config, coach=coach)
    return coach

def create_and_run(file, displaymode):
    coach = create_artificial_neural_network_from_config_file(file)
    coach.run(epochs=coach.config.epochs, display_graph=displaymode, hinton_plot=False)
    # coach.mapping(coach.config.manager.training_cases, number_of_cases=10)
    return coach

## Setup:
displaymode = True

#### TRAINED CASES:
# "one-hot.json"
# "parity.json"
# "segment-counter.json"
file = "glass.json"
coach = None
for arg in sys.argv:
    if arg in ["-r", "remote"]: displaymode=False
    if ".json" in arg:          file=arg

if file:
    coach = create_and_run(file, displaymode)

run = True
while run:
    x = input(">>> ").strip()

    # System:
    if x in ["q", "quit", "exit"]: run = False
    elif x in ["new ann"]: coach = create_and_run(input_json_file(), displaymode)

    if coach:
        # Visuals:
        if x in ["tb", "tensorboard"]: coach.run_tensorboard()
        # TODO: Display biases and weigths. check out display matrix in tftools. note, also imlement in config file
        elif x in ["display hinton history", "dhh"]: coach.display_hinton_graph_from_training_history()
        elif x in ["mapping"]: coach.mapping(coach.config.manager.training_cases, input_number("map batch size="))
        elif x in ["close windows", "close"]: coach.close_all_matplotlib_windows()

        # Stats:
        elif x in ["info","settings", "i"]: coach.config.print_summary()
        elif x == "run more": coach.run_more(input_number("epochs="))
        elif x in "run tests": coach.run_all_tests(renew_session=True, in_top_k=True)
        else: print("Unknown command.")

