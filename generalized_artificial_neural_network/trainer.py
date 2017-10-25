import os
from time import sleep, time
import math
import tensorflow as tf
from numpy.core.multiarray import ndarray

from generalized_artificial_neural_network.network import NeuralNetwork
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration
from generalized_artificial_neural_network.visualizer import VisualizationAdapter


class Trainer():

    def __init__(self, file:str, session:tf.Session=None):

        if not session:
            # Setting up the neural network:
            self.config = NetworkConfiguration(file)
            self.ann = NeuralNetwork(self.config)

            # Session needs to be created after the network is configured.
            self.session = self._create_session()

        # Data to create graphs from:
        self.error_history = []              # History. Each entry is a tuple of (epoch, error)
        self.validation_history = []         # History. Each entry is a tuple of (epoch, error)
        self.monitored_modules_history = []  # History. Each entry is a tuple of (epoch, error)
        self.probed_modules = None
        self.monitored_modules = []
        self.visualizer = VisualizationAdapter(trainer=self)

        # This is a counter over how many cases has run through the network, total.
        self.steps = 0
        self.epochs_trained_on = 0

        # Graphics
        self.graph = None

    def run(self, epochs=None, display_graph:bool=True, hinton_plot:bool=False):
        if self.graph:
            self.graph.figure.clear()
            self.graph = None
        start_time = time()
        epochs = self.config.epochs if not epochs else epochs

        if display_graph:
            from generalized_artificial_neural_network.live_graph import LiveGraph
            self.graph = LiveGraph(graph_title="Error",
                                   x_title="Epochs",
                                   y_title="Error",
                                   epochs=epochs+self.epochs_trained_on)

        self.live_hinton_modules = self.config.hinton_plot

        # Training
        self.train(epochs)

        self.run_all_tests(in_top_k=self.config.in_top_k_test)

        # Closing session:
        self._save_session_params(session=self.session)
        self._close_session(self.session)
        print("\nTime used for this run: " + str(time() - start_time) + " sec")

    def run_more(self, epochs, display_graph:bool=True, hinton_plot:bool=False):
        self._reopen_current_session()
        self.run(epochs, display_graph, hinton_plot)

    def train(self,epochs, average_error=True, live_training_accuracy=True):
        """
        Trains the network on the casemanager training cases.
        :param epochs: Number of times to train on the whole set of cases
        """
        if self.live_hinton_modules:  self.visualizer._initialize_graphics()

        # Getting the cases to run with:
        cases = self.config.manager.get_training_cases()

        # Setting the parameters for the session.run.

        parameters = [self.ann.trainer, self.ann.error]
        trainer = 0; err = 1;  probes = -1; monitored = -1

        if self.config.probe_layers:
            self.probed_modules = tf.summary.merge_all()
            parameters += [self.probed_modules]
            probes = 2

        if self.monitored_modules is not None:
            parameters += [self.monitored_modules]
            monitored = 3 if probes == 2 else 2

        # Scaling minibatch sizes:
        if len(cases) < self.config.mini_batch_size:
            self.config.mini_batch_size = len(cases)-1

        # Looping through epochs. One epoch is a run through all cases.
        for epoch in range(epochs):
            error = 0

            # Looping through each case, running with tensorflow.
            for cases_start in range(0, len(cases), self.config.mini_batch_size):
                # Setting the input and the desired target for this case.:
                feeder_dictionary = self._convert_to_feeder_dictionary(
                    cases[cases_start : (cases_start + self.config.mini_batch_size) ]
                )

                # Actually running:
                results = self.session.run( parameters, feed_dict=feeder_dictionary )

                # Updating variables:
                error += results[err]
                self.steps += 1

            # Updating error history for the graph:
            error = (error / math.ceil(len(cases)/self.config.mini_batch_size)) if average_error else error
            self.error_history.append((epoch+self.epochs_trained_on, error))

            # Perform validation test if interval:
            if epoch % self.config.validation_interval == 0:
                valid_error, valid_accuracy = self.test(cases=self.config.manager.get_validation_cases(), both=True)
                self.validation_history += [(epoch+self.epochs_trained_on, valid_error)]

            # Printing status update:
            if epoch % self.config.display_interval == 0:
                if self.graph:
                    # TODO: Implement other loss function as well. % incorrect cases.
                    self.graph.update(self.error_history, self.validation_history)
                if self.live_hinton_modules:
                    self.visualizer._draw_hinton_graph(results[monitored], epoch, self.monitored_modules)

                # Should turn this off for performance.
                if live_training_accuracy: test_accuracy = self.test(cases=cases, in_top_k=True)
                else: test_accuracy = None

                self._progress_print(epoch, error, valid_accuracy, test_accuracy)
                self.monitored_modules_history += [(epoch, results[monitored])]

        self.epochs_trained_on += epochs
        print("\nTRAINING COMPLETE!")
        print("\tERROR AFTER TRAINING: " + str(self.error_history[-1][1]))

    def test(self, cases:list, renew_session=False, in_top_k=False, both=False):
        if renew_session: self._reopen_current_session()
        # Loading up all data into the feeder dictionary. That way only one call to
        # session.run() is needed for entire test. This is faster.
        target_vectors = []
        feeder_dictionary = self._convert_to_feeder_dictionary(cases, target_vectors=target_vectors)

        # Selecting error function:
        if in_top_k or both:
            if isinstance(target_vectors[0], ndarray):
                target_vectors = [li.tolist() for li in target_vectors]
            labels = [ v.index(1) for v in target_vectors ]
            test_module = [self._create_in_top_k_operator(self.ann.predictor, labels)]

        else: test_module = [self.ann.error]
        if both: test_module += [self.ann.error]

        # Setting the parameters for the session.run.
        parameters = test_module + [self.ann.predictor]

        # Actually running:
        results = self.session.run( parameters, feed_dict=feeder_dictionary )
        if in_top_k: return 100*(results[0] /len(cases))  # results[1] is error. Scaling to fit.
        elif both:   return results[1], 100*(results[0] /len(cases))
        else:        return results[0]               # results[1] is error

    def mapping(self, cases, number_of_cases=10):
        if not self.monitored_modules:
            print("No monitored modules given in json config file. Cannot perform mapping")
            return
        self._reopen_current_session()

        input_vectors = []
        feeder_dictionary = self._convert_to_feeder_dictionary(cases[:number_of_cases], input_vectors=input_vectors)

        # Setting the parameters for the session.run.
        parameters = [self.ann.error, self.ann.predictor, self.monitored_modules]
        results = self.session.run( parameters, feed_dict=feeder_dictionary )

        # Drawing hinton plot:
        self.visualizer._draw_hinton_graph(results[2], number_of_cases, self.monitored_modules)

        # Drawing dendrograms:
        if number_of_cases > 1:
            self.visualizer._draw_dendrograms(features=results[2], labels=input_vectors)

        self._save_session_params(session=self.session)
        self._close_session(self.session)

    def display_weights_and_biases(self, layer=-1, number_of_cases=1, weights=False, biases=False):
        self._reopen_current_session()
        modules = []
        if layer == -1:
            if weights: modules += self.ann.retrieve_all_weights()
            if biases:  modules += self.ann.retrieve_all_biases()
        else:
            if weights: modules += [ self.ann.hidden_layers[layer].weight_matrix ]
            if biases:  modules += [ self.ann.hidden_layers[layer].bias_vector ]

        cases = self.config.manager.training_cases
        input_vectors = []
        feeder_dictionary = self._convert_to_feeder_dictionary(cases[:number_of_cases], input_vectors=input_vectors)

        # Setting the parameters for the session.run.
        parameters = [self.ann.error, self.ann.predictor, modules]
        results = self.session.run( parameters, feed_dict=feeder_dictionary )

        # self.visualizer._draw_hinton_graph(results[2], number_of_cases, modules)
        self.visualizer.display_matrix(modules, results[2])

        self._save_session_params(session=self.session)
        self._close_session(self.session)

    def run_all_tests(self, renew_session=False, in_top_k=True):
        if renew_session: self._reopen_current_session()

        # Running tests:
        training_score = self.test(self.config.manager.get_training_cases(), in_top_k=in_top_k)
        validation_score = self.test(self.config.manager.get_validation_cases(), in_top_k=in_top_k)
        testing_score = self.test(self.config.manager.get_testing_cases(), in_top_k=in_top_k)

        # Print stats:
        print("\nPERFORMING TESTS:")
        print("\tTRAINING CASES:   " + str(training_score) + " % CORRECT")
        print("\tVALIDATION CASES: " + str(validation_score) + " % CORRECT")
        print("\tTESTING CASES:    " + str(testing_score) + " % CORRECT")

    def _convert_to_feeder_dictionary(self, cases, input_vectors=None, target_vectors=None):
        input_vectors  = [] if input_vectors  is None else input_vectors
        target_vectors = [] if target_vectors is None else target_vectors
        for case in cases:
            input_vectors += [case[0]]
            target_vectors += [case[1]]
        return {self.ann.input: input_vectors, self.ann.target: target_vectors}

    def monitor_module(self, module_index, type='wgt'):
        """
        Adds a module from the network (i.e. a weight matrix, input vector, output vector or bias vector)
        to a list of monitored modules. The monitored modules will be run with session.run() to get live
        data about the module.
        NB! All modules are displayed using the hinton graph.
        """
        self.monitored_modules.append(self.ann.hidden_layers[module_index].get_variable(type))

    def _progress_print(self, epoch, error, valid_accurracy=None, test_accurracy=None):
        output = "Epoch=" + "0"*(len(str(self.config.epochs)) - len(str(epoch))) + str(epoch) + "    " \
                 "Error=" + str(error) + "    " \
                 "Validation error=" + (str(self.validation_history[-1][1]) if self.validation_history else "0")

        if valid_accurracy and test_accurracy:
            output += "    Validation accuracy="+str(valid_accurracy)+"%    Training Accuracy="+str(test_accurracy)+"%"
        print(output)

    def _create_in_top_k_operator(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    # TODO: Separate all session handling into external class SessionHandler.
    # TODO: implement saving sessions.

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

    def _close_session(self, session: tf.Session):
        session.probe_stream.close()
        session.close()

    def _copy_session(self, session):
        session = session if session else self.session
        copied_session = tf.Session()
        copied_session.probe_stream = session.probe_stream
        copied_session.probe_stream.reopen()
        copied_session.viewdir = session.viewdir
        return copied_session

    def _reopen_current_session(self):
        self.session = self._copy_session(self.session)  # Open a new session with same tensorboard stuff
        self.session.run(tf.global_variables_initializer())
        self._restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def _save_session_params(self, spath='netsaver/my_saved_session', session=None, step=0):
        session = session if session else self.current_session
        state_vars = []
        for m in self.ann.hidden_layers:
            vars = [m.get_variable('wgt'), m.get_variable('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def _restore_session_params(self, path=None, session=None):
        spath = path if path else self.saved_state_path
        session = session if session else self.session
        self.state_saver.restore(session, spath)

    def run_tensorboard(self, session: tf.Session =None):
        session = session if session else self.session
        session.probe_stream.flush()
        session.probe_stream.close()
        os.system('tensorboard --logdir=' + session.viewdir)
