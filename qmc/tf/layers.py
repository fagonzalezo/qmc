"""
Layers implementing quantum feature maps, measurements and
utilities.
"""

import numpy as np
import tensorflow as tf
from typeguard import typechecked
from sklearn.kernel_approximation import RBFSampler



##### Quantum Feature Map Layers


class QFeatureMapSmp(tf.keras.layers.Layer):
    """Quantum feature map using soft max probabilities.
    input values are asummed to be between 0 and 1.

    Input shape:
        (batch_size, dim1)
    Output shape:
        (batch_size, dim ** dim1)
    Arguments:
        dim: int. Number of dimensions to represent each value
        beta: parameter beta of the softmax function
    """

    @typechecked
    def __init__(
            self,
            dim: int = 2,
            beta: float = 4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.beta = beta


    def build(self, input_shape):
        self.points = tf.reshape(tf.linspace(0., 1., self.dim),
                                 len(input_shape)*[1] + [self.dim])
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        vals = tf.expand_dims(inputs, axis=-1) # shape (..., n, 1)
        dists = (self.points - vals) ** 2 # shape (..., n, dim)
        sm = tf.exp(-dists * self.beta) # shape (..., n, dim)
        sums = tf.math.reduce_sum(sm, axis=-1) # shape (..., n)
        sm = sm / tf.expand_dims(sums, axis=-1) # shape (..., n, dim)
        amp = tf.sqrt(sm) # shape (..., n, dim)
        b_size = tf.shape(amp)[0]
        t_psi = amp[:, 0, :]
        for i in range(1, amp.shape[1]):
            t_psi = tf.einsum('...i, ...j->...ij', t_psi, amp[:, i, :])
            t_psi = tf.reshape(t_psi, (b_size, -1))
        return t_psi

    def get_config(self):
        config = {
            "dim": self.dim,
            "beta": self.beta
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim ** input_shape[1])

class QFeatureMapOneHot(tf.keras.layers.Layer):
    """Quantum feature map using one-hot encoding.
    input values are indices, with 0<index<num_classes

    Input shape:
        (batch_size, dim)
    Output shape:
        (batch_size, num_classes ** dim)
    Arguments:
        num_classes: int. Number of dimensions to represent each value
    """

    @typechecked
    def __init__(
            self,
            num_classes: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `QFeatureMapOneHot` layer should be called '
                             'on a tensor with shape (batch_size, dim)')
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        out = tf.keras.backend.one_hot(inputs, self.num_classes)
        b_size = tf.shape(out)[0]
        out = tf.reshape(out, (b_size, -1))
        return out

    def get_config(self):
        config = {
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes ** input_shape[1])

class QFeatureMapRFF(tf.keras.layers.Layer):
    """Quantum feature map using random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed. 
    """

    @typechecked
    def __init__(
            self,
            dim: int = 100,
            gamma: float = 1,
            random_state=None, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma, 
            n_components=self.dim,
            random_state=self.random_state)
        X = np.zeros(shape=input_shape.as_list())
        rbf_sampler.fit(X)
        self.rff_weights = tf.constant(
            rbf_sampler.random_weights_,
            dtype=tf.float32)
        self.offset = tf.constant(
            rbf_sampler.random_offset_,
            dtype=tf.float32)
        self.built = True

    def call(self, inputs):
        vals = tf.matmul(inputs, self.rff_weights) + self.offset
        vals = tf.cos(vals)
        vals = vals * tf.sqrt(2. / self.dim)
        norms = tf.linalg.norm(vals, axis=1)
        psi = vals / tf.expand_dims(norms, axis=-1)
        return psi

    def get_config(self):
        config = {
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

##### Quantum Measurement layers

class QMeasurement(tf.keras.layers.Layer):
    """Quantum measurement layer.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_y: int. the dimension of the output state
    """

    @typechecked
    def __init__(
            self,
            dim_y: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_y = dim_y

    def build(self, input_shape):
        self.rho = self.add_weight(
            "rho",
            shape=(input_shape[1], self.dim_y, input_shape[1], self.dim_y),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        oper = tf.einsum(
            '...i,...j->...ij',
            inputs, tf.math.conj(inputs),
            optimize='optimal') # shape (b, nx, nx)
        rho_res = tf.einsum(
            '...ik, klmn, ...mo -> ...ilon',
            oper, self.rho, oper,
            optimize='optimal')  # shape (b, nx, ny, nx, ny)
        trace_val = tf.einsum('...ijij->...', rho_res, optimize='optimal') # shape (b)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        rho_res = rho_res / trace_val
        rho_y = tf.einsum('...ijik->...jk', rho_res, optimize='optimal') # shape (b, ny, ny)
        return rho_y

    def get_config(self):
        config = {
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

##### Util layers

class CrossProduct(tf.keras.layers.Layer):
    """Calculates the cross product of 2 inputs.

    Input shape:
        A list of 2 tensors [t1, t2] with shapes
        (batch_size, n) and (batch_size, m)
    Output shape:
        (batch_size, n, m)
    Arguments:
    """

    @typechecked
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)


    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `CrossProduct` layer should be called '
                             'on a list of 2 inputs.')
        if len(input_shape[0]) > 11 or len(input_shape[1]) > 11:
            raise ValueError('Input tensors cannot have more than '
                             '11 dimensions.')
        idx1 = 'abcdefghij'
        idx2 = 'klmnopqrst'
        self.eins_eq = ('...' + idx1[:len(input_shape[0]) - 1] + ',' +
                        '...' + idx2[:len(input_shape[1]) - 1] + '->' +
                        '...' + idx1[:len(input_shape[0]) - 1] +
                        idx2[:len(input_shape[1]) - 1])
        self.built = True

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `CrossProduct` layer should be called '
                             'on exactly 2 inputs')
        cp = tf.einsum(self.eins_eq,
                       inputs[0], inputs[1], optimize='optimal')
        return cp

    def compute_output_shape(self, input_shape):
        return (input_shape[0][1], input_shape[1][1])

class DensityMatrix2Dist(tf.keras.layers.Layer):
    """Extracts a probability distribution from a density matrix.

    Input shape:
        A tensor with shape (batch_size, n, n)
    Output shape:
        (batch_size, n)
    Arguments:
    """

    @typechecked
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)


    def build(self, input_shape):
        if len(input_shape) != 3 or input_shape[1] != input_shape[2]:
            raise ValueError('A `DensityMatrix2Dist` layer should be '
                             'called with a tensor of shape '
                             '(batch_size, n, n)')
        self.built = True

    def call(self, inputs):
        if len(inputs.shape) != 3 or inputs.shape[1] != inputs.shape[2]:
            raise ValueError('A `DensityMatrix2Dist` layer should be '
                             'called with a tensor of shape '
                             '(batch_size, n, n)')
        cp = tf.einsum('...ii->...i', inputs, optimize='optimal')
        return cp

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[1])
