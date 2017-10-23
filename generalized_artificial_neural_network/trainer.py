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

            # Perform validation test if interval:
            if epoch % self.config.validation_interval == 0:
                self.validation_history += [(epoch, self.test(cases=self.config.manager.get_validation_cases()))]


            # Printing status update:
            if epoch % self.config.display_interval == 0:
                if self.graph:
                    self.graph.update(self.error_history, self.validation_history)
                self.progress_print(epoch, error)

    def test(self, cases:list, in_top_k=False):
        input_vectors = []
        target_vectors = []
        for case in cases:
            # Setting the input and the desired target for this case.:
            input_vectors += [case[0]]
            target_vectors += [case[1]]

        feeder_dictionary = {self.ann.input: input_vectors, self.ann.target: target_vectors}

        # Selecting error function:
        if not in_top_k:
            test_module = self.ann.predictor
        else:
            labels = [ v.index(1) for v in target_vectors ]
            test_module = self._create_in_top_k_operator(self.ann.predictor, labels)

        # Setting the parameters for the session.run.
        parameters = [test_module, self.ann.error]

        # Actually running:
        results = self.session.run( parameters, feed_dict=feeder_dictionary )

        # print("TEST ERROR: " + str(results[1]))
        return results[1]

    def progress_print(self, epoch, error):

        print("Epoch=" + "0"*(len(str(self.config.epochs)) - len(str(epoch))) + str(epoch) + "    "
              "Error=" + str(error) + "    "
              "Validation=" + (str(self.validation_history[-1][1]) if self.validation_history else "0"))

    def _create_in_top_k_operator(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

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

    coach = Trainer(configuration_file, display_graph=False)
    coach.train(epochs=coach.config.epochs)
    coach.test(coach.config.manager.get_testing_cases(), in_top_k=False)

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