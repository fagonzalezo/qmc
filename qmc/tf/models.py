'''
Quantum Measurement Classfiication Models
'''

import tensorflow as tf
from . import layers

class QMClassifier(tf.keras.Model):
    """
    A Quantum Measurement Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMClassifier, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

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
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMClassifier, self).fit(*args, **kwargs)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):
        return self.qm.rho

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
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
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
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
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class ComplexQMClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Classifier model trainable using
    gradient descent with complex terms.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(ComplexQMClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapComplexRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.ComplexQMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dm2dist = layers.ComplexDensityMatrix2Dist()
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
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMDensity(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
    """
    def __init__(self, fm_x, dim_x):
        super(QMDensity, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.qmd = layers.QMeasureDensity(dim_x)
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        return probs

    @tf.function
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x = data
        rho = self.call_train(x)
        self.qmd.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMDensity, self).fit(*args, **kwargs)
        self.qmd.weights[0].assign(self.qmd.weights[0] / self.num_samples)
        return result

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class QMDensitySGD(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation modeltrainable using
    gradient descent.
    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, num_eig=0, gamma=1, random_state=None):
        super(QMDensitySGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
        self.num_eig = num_eig
        self.dim_x = dim_x
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        self.add_loss(-tf.reduce_sum(tf.math.log(probs)))
        return probs

    def set_rho(self, rho):
        return self.qmd.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class DMKDClassifier(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
        num_classes: int number of classes
    """
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(DMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensity(dim_x))
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=tf.zeros((num_classes,)),
            trainable=False
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors / 
            tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    @tf.function
    def call_train(self, x, y):
        if not self.qmd[0].built:
            self.call(x)
        psi = self.fm_x(x) # shape (bs, dim_x)
        rho = self.cp([psi, tf.math.conj(psi)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = ohy * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos

    def train_step(self, data):
        x, y = data
        rhos = self.call_train(x, y)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign_add(rhos[i])
        return {}

    def fit(self, *args, **kwargs):
        result = super(DMKDClassifier, self).fit(*args, **kwargs)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] /
                                          self.num_samples[i])
        return result

    def get_rhos(self):
        weights = [qmd.weights[0] for qmd in self.qmd]
        return weights

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}

class ComplexDMKDClassifier(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model with complex terms.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
        num_classes: int number of classes
    """
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(ComplexDMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.ComplexQMeasureDensity(dim_x))
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=tf.zeros((num_classes,)),
            trainable=False
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        return posteriors

    @tf.function
    def call_train(self, x, y):
        if not self.qmd[0].built:
            self.call(x)
        psi = self.fm_x(x) # shape (bs, dim_x)
        rho = self.cp([psi, tf.math.conj(psi)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = tf.cast(ohy, tf.complex64) * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos

    def train_step(self, data):
        x, y = data
        rhos = self.call_train(x, y)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign_add(rhos[i])
        return {}

    def fit(self, *args, **kwargs):
        result = super(ComplexDMKDClassifier, self).fit(*args, **kwargs)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] /
                                          tf.cast(self.num_samples[i], tf.complex64))
        return result

    def get_rhos(self):
        weights = [qmd.weights[0] for qmd in self.qmd]
        return weights

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}

class DMKDClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes: number of classes
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, dim_x, num_classes, num_eig=0, gamma=1, random_state=None):
        super(DMKDClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensityEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors / 
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class ComplexDMKDClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent using complex random fourier features.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes: number of classes
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, dim_x, num_classes, 
                 num_eig=0, gamma=1, random_state=None):
        super(ComplexDMKDClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapComplexRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.ComplexQMeasureDensityEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors / 
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressor(tf.keras.Model):
    """
    A Quantum Measurement Regression model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMRegressor, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dmregress = layers.DensityMatrixRegression()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    @tf.function
    def call_train(self, x, y):
        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMRegressor, self).fit(*args, **kwargs)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):
        return self.weights[2]

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressorSGD(tf.keras.Model):
    """
    A Quantum Measurement Regressor model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMRegressorSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dmregress = layers.DensityMatrixRegression()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class ComplexQMRegressorSGD(tf.keras.Model):
    """
    A Quantum Measurement Regressor model trainable using
    gradient descent with complex terms.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(ComplexQMRegressorSGD, self).__init__()
        self.fm_x = layers.QFeatureMapComplexRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.ComplexQMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dmregress = layers.ComplexDensityMatrixRegression()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}