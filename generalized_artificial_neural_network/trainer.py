import os
from time import sleep, time

import tensorflow as tf

from generalized_artificial_neural_network.network import NeuralNetwork
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration



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

        # Graphics
        self.hinton_figures = []     # list of matplotlib.pyplot.Figure
        self.dendrogram_figures = [] # list of matplotlib.pyplot.Figure

    def run(self, epochs=None, display_graph:bool=True, hinton_plot:bool=False):
        self.graph = None
        start_time = time()
        epochs = self.config.epochs if not epochs else epochs

        if display_graph:
            from generalized_artificial_neural_network.live_graph import LiveGraph
            self.graph = LiveGraph(graph_title="Error", x_title="Epochs", y_title="Error", epochs=epochs)

        self.live_hinton_modules = hinton_plot

        # Training
        self.train(epochs)

        self.run_all_tests(in_top_k=True)

        # Closing session:
        self._save_session_params(session=self.session)
        self._close_session(self.session)
        print("\nTime used for this run: " + str(time() - start_time) + " sec")

    def run_more(self, epochs, display_graph:bool=True, hinton_plot:bool=False):
        self._reopen_current_session()
        self.run(epochs, display_graph, hinton_plot)

    def train(self, epochs):
        """
        Trains the network on the casemanager training cases.
        :param epochs: Number of times to train on the whole set of cases
        """
        if self.live_hinton_modules:  self._initialize_graphics()

        # This is a counter over how many cases has run through the network, total.
        steps = 0

        # Getting the cases to run with:
        cases = self.config.manager.get_training_cases()

        # Setting the parameters for the session.run.
        self.probed_modules = tf.summary.merge_all()

        parameters = [self.ann.trainer, self.ann.error, self.probed_modules]
        trainer = 0; err = 1;  probes = 2; monitored = 3

        if self.monitored_modules is not None:parameters += [self.monitored_modules]

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
                if self.live_hinton_modules:
                    self.display_hinton_module(results[monitored], epoch)
                self._progress_print(epoch, error)
                self.monitored_modules_history += [(epoch, results[monitored])]
        print("\nTRAINING COMPLETE!")
        print("\tERROR AFTER TRAINING: " + str(self.error_history[-1][1]))

    def test(self, cases:list, renew_session=False, in_top_k=False):
        if renew_session: self._reopen_current_session()
        # Loading up all data into the feeder dictionary. That way only one call to
        # session.run() is needed for entire test. This is faster.
        target_vectors = []
        feeder_dictionary = self._convert_to_feeder_dictionary(cases, target_vectors=target_vectors)

        # Selecting error function:
        if in_top_k:
            labels = [ v.index(1) for v in target_vectors ]
            test_module = self._create_in_top_k_operator(self.ann.predictor, labels)
        else: test_module = self.ann.error

        # Setting the parameters for the session.run.
        parameters = [test_module, self.ann.predictor]

        # Actually running:
        results = self.session.run( parameters, feed_dict=feeder_dictionary )
        if in_top_k: return 100*(results[0] /len(cases))  # results[1] is error. Scaling to fit.
        else:        return results[0]               # results[1] is error

    def mapping(self, cases, number_of_cases=10):
        self._reopen_current_session()

        input_vectors = []
        feeder_dictionary = self._convert_to_feeder_dictionary(cases[:number_of_cases], input_vectors=input_vectors)

        # Setting the parameters for the session.run.
        parameters = [self.ann.error, self.ann.predictor, self.monitored_modules]
        results = self.session.run( parameters, feed_dict=feeder_dictionary )

        # Drawing hinton plot:
        self._draw_hinton_graph(results[2], number_of_cases)

        # Drawing dendrograms:
        self._draw_dendrograms(input_vectors, results[2])

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

    def display_hinton_graph_from_training_history(self):
        self._initialize_graphics()
        from downing_code.tflowtools import hinton_plot
        for history in self.monitored_modules_history:
            self._draw_hinton_graph(history[1], history[0], hinton_plot)

    def _draw_hinton_graph(self, graph_results, epoch, imported_method=None):
        if not imported_method:
            from downing_code.tflowtools import hinton_plot
            self._initialize_graphics()
        else: hinton_plot = imported_method
        if self.hinton_figures and graph_results:
            # Local import to be able to run on server.
            for i in range(len(self.monitored_modules)):
                hinton_plot(
                    matrix=graph_results[i],
                    fig=self.hinton_figures[i],
                    title=self.monitored_modules[i].name + " @ epoch=" + str(epoch))

    def _draw_dendrograms(self, features, labels):
        from downing_code.tflowtools import dendrogram
        import matplotlib.pyplot as PLT
        for i, monitored in enumerate(labels):
            module_name = self.monitored_modules[i].name

            if any(word in module_name for word in ["in", "bias", "out"]):
                self.dendrogram_figures.append(PLT.figure())
                dendrogram(features, labels=monitored, figure=self.dendrogram_figures[-1], title=module_name)

    def _initialize_graphics(self):
        """ Creates the list of "matplotlib.pyplot.Figure" to be used with the hinton diagrams. """
        import matplotlib.pyplot as PLT
        for i in range(len(self.monitored_modules)):
            self.hinton_figures.append(PLT.figure())

    def _progress_print(self, epoch, error):
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

    def close_all_matplotlib_windows(self):
        import matplotlib.pyplot as PLT
        PLT.close('all')

    def run_tensorboard(self, session: tf.Session =None):
        session = session if session else self.session
        session.probe_stream.flush()
        session.probe_stream.close()
        os.system('tensorboard --logdir=' + session.viewdir)
