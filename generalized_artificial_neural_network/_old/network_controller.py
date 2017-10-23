import math

import numpy as np
import tensorflow as tf

import downing_code.tflowtools as TFT
from generalized_artificial_neural_network._old.neural_network import NeuralNetwork
from generalized_artificial_neural_network.live_graph import LiveGraph
from generalized_artificial_neural_network.network_configuration import NetworkConfiguration


class NetworkController():

    def __init__(self, configuration: NetworkConfiguration):
        self.casemanager = configuration.manager
        self.graph = None
        self.net = NeuralNetwork(configuration)
        self.validation_history = []
        self.validation_interval = configuration.validation_interval
        self.show_interval = configuration.display_interval


    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):

            error = 0
            step = self.net.global_training_step + i
            gvars = [self.net.error] + self.net.grabvars
            mbs = self.net.minibatch_size
            ncases = len(cases)
            nmb = math.ceil(ncases/mbs)

            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.net.input: inputs, self.net.target: targets}
                _, grabvals, __ = self.run_one_step(
                    [self.net.trainer],
                    gvars,
                    self.net.probes,
                    session=sess,
                    feed_dict=feeder,
                    step=step,
                    show_interval=self.net.show_interval)
                error += grabvals[0]

            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess)
            if self.show_interval and (step % self.show_interval == 0):
                self.graph.update(self.error_history, self.validation_history)

        # TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
        #                           title="", fig=not(continued))
        self.net.global_training_step += epochs

    def do_testing(self,sess,cases,msg='Testing',bestk=1):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.net.input: inputs, self.net.target: targets}
        self.test_func = self.net.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(
                self.net.predictor,
                [TFT.one_hot_to_int(list(v)) for v in targets],
                k=bestk)

        testres, grabvals, _ = self.run_one_step(
            self.test_func,
            self.net.grabvars,
            self.net.probes,
            session=sess,
            feed_dict=feeder,
            show_interval=self.show_interval)

        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * testres / len(cases)))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None


    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.net.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session

        self.graph = LiveGraph(x_title="Epoch",y_title="Error", graph_title="Error vs validation", epochs=epochs)
        self.do_training(session,self.casemanager.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess):
        cases = self.casemanager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing')

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.casemanager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess):
        self.do_testing(sess,self.casemanager.get_training_cases(),msg='Total Training')

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)

        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)

        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("\t" + names[i] + ": ", end="")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,fig=self.net.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            else:
                print(v, end=('\n'))

    def run(self, epochs=100, sess=None, continued=False):
        # PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session)
        self.testing_session(sess=self.current_session)
        self.close_current_session()
        # PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.net.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=False)

