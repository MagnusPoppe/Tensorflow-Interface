import os
from time import sleep

import tensorflow as tf

from generalized_artificial_neural_network.network import NeuralNetwork
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration


class Trainer():

    def __init__(self, file:str, session:tf.Session=None, display_graph:bool=True):

        if not session:

            self.config = NetworkConfiguration(file)
            self.ann = NeuralNetwork(self.config)

            # Needs to be set after the network is configured.
            self.session = self._create_session()

        # Data to create graphs from:
        self.error_history = []
        self.validation_history = []

        if display_graph:
            from generalized_artificial_neural_network.live_graph import LiveGraph
            self.graph = LiveGraph(graph_title="Error", x_title="Epochs", y_title="Error", epochs=self.config.epochs)
        else: self.graph = None

    def train(self, epochs):
        """
        Trains the network on the casemanager training cases.
        :param epochs: Number of times to train on the whole set of cases
        """
        # This is a counter over how many cases has run through the network, total.
        steps = 0

        # Getting the cases to run with:
        cases = self.config.manager.get_training_cases()

        # Looping through epochs. One epoch is a run through all cases.
        for epoch in range(epochs):
            error = 0

            # Looping through each case, running with tensorflow.
            for case in cases:
                # Setting the input and the desired target for this case.:
                input_vector  = case[0]
                target_vector = case[1]
                feeder_dictionary = {self.ann.input: [input_vector], self.ann.target: [target_vector]}

                # Setting the parameters for the session.run.
                parameters = [self.ann.trainer, self.ann.error]

                # Actually running:
                results = self.session.run( parameters, feed_dict=feeder_dictionary )

                # Updating variables:
                error += results[1]
                steps += 1

            # Updating error history for the graph:
            self.error_history.append((epoch, error))

            # Printing status update:

            # TODO: Validation interval?
            if epoch % self.config.display_interval == 0:
                if self.graph:
                    self.graph.update(self.error_history)
                print("Epoch="+str(epoch)+"    Error="+str(error))

    def test(self, cases:list):
        for case in cases:
            # Setting the input and the desired target for this case.:
            input_vector = case[0]
            target_vector = case[1]
            feeder_dictionary = {self.ann.input: [input_vector], self.ann.target: [target_vector]}

            # Setting the parameters for the session.run.
            parameters = [self.ann.predictor, self.ann.error]

            # Actually running:
            results = self.session.run( parameters, feed_dict=feeder_dictionary )

            print("ERROR: " + str(results[1]))

    def _create_session(self, directory='probeview') -> tf.Session:
        # Clearing the output folders for previous output:
        os.system('rm ' + directory + '/events.out.*')

        # Creating session:
        session = tf.Session()

        # Create a probe stream and attach to the session (for use with tensorboard)
        session.probe_stream = tf.summary.FileWriter(directory, session.graph, flush_secs=120, max_queue=10)
        session.viewdir = directory  # add a second slot, viewdir, to the session

        # Initializing variables:
        session.run(tf.global_variables_initializer())
        return session


if __name__ == '__main__':
    file = "bit-counter.json" #"glass.json" #"one-hot.json"
    configuration_file = os.path.join("configurations", file)

    coach = Trainer(configuration_file, display_graph=True)
    coach.train(epochs=coach.config.epochs)
    coach.test(coach.config.manager.get_testing_cases())

    print("RUN COMPLETE!")
    print("Error after training: " + str(coach.error_history[-1][1]))

    exit(0)

    run = True
    waited = 0
    while run:
        x = input(">>> ")

        if x in ["q", "quit"]:
            run = False

        elif x == "run more":
            pass

        elif x in "run tests":
            coach.test(coach.config.manager.get_testing_cases())