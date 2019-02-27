import tensorflow as tf
import numpy as np
import datetime
from som import SOM

input_data = np.array([[2., 1., 1., 1., 3.],
                       [2., 1., 0., 0., 3.],
                       [2., 1., 0., 1., 2.],
                       [2., 1., 0., 1., 3.],
                       [4., 0., 0., 0., 2.],
                       [4., 0., 1., 0., 1.],
                       [4., 0., 1., 0., 3.],
                       [4., 0., 1., 0., 2.],
                       [0., 0., 1., 0., 3.],
                       [0., 0., 1., 0., 2.]])

class GHSOM(object):

    #To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, tau1=None, tau2=None):
        #Assign required variables first
        self._m = m
        self._n = n
        if tau1 is None:
            tau1 = 0.1
        else:
            tau1 = float(tau1)
        if tau2 is None:
            tau2 = 0.1
        else:
            tau2 = float(tau2)

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # initial tensorflow variable
            self.input_data_tf = tf.placeholder("float")

            self.mqe = tf.reduce_mean(
                        tf.sqrt(
                            tf.reduce_sum(
                                tf.pow(
                                    tf.subtract(
                                        self.input_data_tf,
                                        tf.stack([tf.reduce_mean(self.input_data_tf, 0) for i in range(dim)])
                                    )
                                ,2)
                            , 1, keep_dims=True)
                        )
                    , 0)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init = tf.global_variables_initializer()
            self._sess.run(init)


    def train(self, input_vect):
        print(self._sess.run(self.mqe, feed_dict={self.input_data_tf: input_vect}))



ghsom = GHSOM(2, 2, 10)
ghsom.train(input_data)
