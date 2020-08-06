'''
Quantum Measurement Classfiication Models
'''

import tensorflow as tf
import qmc.tf.layers as layers

class QMClassifier(tf.keras.Model):
    """
    A Quantum Measurement Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_y: dimension of the output representation
    """
    def __init__(self, fm_x, fm_y, dim_y):
        super(QMClassifier, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasurement(dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    @tf.function
    def call_train(self, x, y):
        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0) / num_samples
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def get_config(self):
        config = {
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}
