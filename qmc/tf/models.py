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


class QMClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, gamma=1, random_state=None):
        super(QMClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasurementEig(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    def set_rho(self, rho):
        self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}
