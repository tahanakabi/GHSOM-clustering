import tensorflow as tf
import numpy as np
import datetime
from .som import SOM

class GHSOM(object):
    #To check if the SOM has been trained
    _trained = False

    #To check if mqe0 is calculated
    _calculated_mqe0 = False


    def __init__(self, m, n, dim, row_num, input_data, tau1, tau2, alpha=None, sigma=None):

        # set ghsom initial som map size
        self._m = m
        self._n = n
        self._row_num = row_num

        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)

        if tau1 is None:
            tau1 = 0.5
        else:
            tau1 = float(tau1)

        if tau2 is None:
            tau2 = 0.5
        else:
            tau2 = float(tau2)



        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # #The input data training matrix
            self._input_data = tf.placeholder("float")


            self._mqe = self._cal_mqe0()

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

            print(self._sess.run(self._mqe,feed_dict={self._input_data: input_data }))

    # cal_mqe0
    def _cal_mqe0(self):
        return tf.reduce_mean(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.pow(
                                tf.subtract(
                                    self._input_data,
                                    tf.stack([tf.reduce_mean(self._input_data, 0) for i in range(self._row_num)])
                                )
                            ,2)
                        , 1, keep_dims=True)
                    )
                , 0)

    # train ghsom
    def train(self, input_data):
        self._sess.run(self._mqe,
                       feed_dict={self._input_data: input_data })
