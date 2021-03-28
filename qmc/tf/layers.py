"""
Layers implementing quantum feature maps, measurements and
utilities.
"""

import numpy as np
import tensorflow as tf
from typeguard import typechecked
from sklearn.kernel_approximation import RBFSampler
from . import _RBFSamplerORF



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
        out = tf.one_hot(tf.cast(inputs, dtype=tf.int32),
                         self.num_classes, on_value=1., off_value=0.)
        b_size = tf.shape(out)[0]
        t_psi = out[:, 0, :]
        for i in range(1, out.shape[1]):
            t_psi = tf.einsum('...i,...j->...ij', t_psi, out[:, i, :])
            t_psi = tf.reshape(t_psi, (b_size, -1))
        return t_psi

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
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """

    @typechecked
    def __init__(
            self,
            input_dim: int,
            dim: int = 100,
            gamma: float = 1,
            random_state=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=True,
            name="rff_weights")
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=True,
            name="offset")
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
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

class QFeatureMapORF(tf.keras.layers.Layer):
    """Quantum feature map using Orthogonal Random Features.
    Uses `ORFSampler` from sklearn to approximate an RBF kernel using
    random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """

    @typechecked
    def __init__(
            self,
            input_dim: int,
            dim: int = 100,
            gamma: float = 1,
            random_state=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        rbf_sampler = _RBFSamplerORF.RBFSamplerORF(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=True,
            name="rff_weights")
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=True,
            name="offset")
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
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)



class QFeatureMapComplexRFF(tf.keras.layers.Layer):
    """Quantum feature map including the complex part of random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    complex random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """

    @typechecked
    def __init__(
            self,
            input_dim: int,
            dim: int = 100,
            gamma: float = 1,
            random_state=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=True,
            name="rff_weights")
        self.built = True

    def call(self, inputs):
        vals = tf.matmul(inputs, self.rff_weights)
        vals = tf.complex(tf.cos(vals), tf.sin(vals))
        vals = vals * tf.cast(tf.sqrt(1. / self.dim), tf.complex64)
        norms = tf.linalg.norm(vals, axis=1)
        psi = vals / tf.expand_dims(norms, axis=-1)
        return psi

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

##### Quantum Measurement layers

class QMeasureClassif(tf.keras.layers.Layer):
    """Quantum measurement layer for classification.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input  state
        dim_y: int. the dimension of the output state
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            dim_y: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.rho = self.add_weight(
            "rho",
            shape=(self.dim_x, self.dim_y, self.dim_x, self.dim_y),
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
            optimize='optimal')  # shape (b, nx, ny, ny, nx)
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
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

class QMeasureClassifEig(tf.keras.layers.Layer):
    """Quantum measurement layer for classification.
    Represents the density matrix using a factorization:

    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is ameanable to gradient-based learning.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            dim_y: int = 2,
            num_eig: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y
        if num_eig < 1:
            num_eig = dim_x * dim_y
        self.num_eig = num_eig

    def build(self, input_shape):
        if input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.eig_vec = self.add_weight(
            "eig_vec",
            shape=(self.dim_x * self.dim_y, self.num_eig),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        eig_vec = tf.reshape(eig_vec, (self.dim_x, self.dim_y, self.num_eig))
        eig_vec_y = tf.einsum('...i,ijk->...jk', inputs,eig_vec, optimize='optimal') # shape (b, ny, ne)
        eig_val_sr = tf.sqrt(eig_val)
        eig_val_sr = tf.expand_dims(eig_val_sr, axis=0)
        eig_val_sr = tf.expand_dims(eig_val_sr, axis=0)
        eig_vec_y = eig_vec_y * eig_val_sr
        rho_y = tf.matmul(eig_vec_y, eig_vec_y, adjoint_b=True)
        trace_val = tf.einsum('...jj->...', rho_y, optimize='optimal') # shape (b)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        rho_y = rho_y / trace_val
        return rho_y

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_y, dim_x, dim_y)
        """
        if (len(rho.shape.as_list()) != 4 or
                rho.shape[0] != self.dim_x or
                rho.shape[2] != self.dim_x or
                rho.shape[1] != self.dim_y or
                rho.shape[3] != self.dim_y):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_y},'
                f' {self.dim_x}, {self.dim_y})')
        if not self.built:
            self.build((None, self.dim_x))
        rho_prime = tf.reshape(
            rho,
            (self.dim_x * self.dim_y, self.dim_x * self.dim_y,))
        e, v = tf.linalg.eigh(rho_prime)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

class ComplexQMeasureClassifEig(tf.keras.layers.Layer):
    """Quantum measurement layer for classification.
    Represents the density matrix with complex values using a factorization:

    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is amenable to gradient-based learning.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            dim_y: int = 2,
            num_eig: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y
        if num_eig < 1:
            num_eig = dim_x * dim_y
        self.num_eig = num_eig

    def build(self, input_shape):
        if input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        with tf.device('cpu:0'):
            self.eig_vec = self.add_weight(
                "eig_vec",
                shape=(self.dim_x * self.dim_y, self.num_eig),
                dtype=tf.complex64,
                initializer=complex_initializer(tf.random_normal_initializer),
                trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.complex64)
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
            tf.cast(tf.linalg.diag(tf.sqrt(eig_val)), tf.complex64))
        rho_h = tf.reshape(
            rho_h,
            (self.dim_x, self.dim_y, self.num_eig))
        rho_h = tf.einsum(
            '...k, klm -> ...lm',
            inputs, rho_h,
            optimize='optimal')
        rho_y = tf.einsum(
            '...ik, ...jk -> ...ij',
            rho_h, tf.math.conj(rho_h),
            optimize='optimal')
        trace_val = tf.einsum('...ii->...', rho_y, optimize='optimal')
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        rho_y = rho_y / trace_val
        return rho_y

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_y, dim_x, dim_y)
        """
        if (len(rho.shape.as_list()) != 4 or
                rho.shape[0] != self.dim_x or
                rho.shape[2] != self.dim_x or
                rho.shape[1] != self.dim_y or
                rho.shape[3] != self.dim_y):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_y},'
                f' {self.dim_x}, {self.dim_y})')
        if not self.built:
            self.build((None, self.dim_x))
        rho_prime = tf.reshape(
            rho, 
            (self.dim_x * self.dim_y, self.dim_x * self.dim_y,))
        e, v = tf.linalg.eigh(rho_prime)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

class QMeasureDMClassif(tf.keras.layers.Layer):
    """Quantum measurement layer for classification.
    Receives as input a density matrix.

    Input shape:
        (batch_size, dim_x, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input  state
        dim_y: int. the dimension of the output state
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            dim_y: int = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y

    def build(self, input_shape):
        if ((not input_shape[1] is None and input_shape[1] != self.dim_x)  
            or (not input_shape[2] is None and input_shape[2] != self.dim_x)):
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x}, {self.dim_x})')
        self.rho = self.add_weight(
            "rho",
            shape=(self.dim_x, self.dim_y, self.dim_x, self.dim_y),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        rho_res = tf.einsum(
            '...ik, klmn, ...mo -> ...ilon',
            inputs, self.rho, inputs,
            optimize='optimal')  # shape (b, nx, ny, ny, nx)
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
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

class QMeasureDMClassifEig(tf.keras.layers.Layer):
    """Quantum measurement layer for classification.
    Receives as input a density matrix.
    Represents the internal density matrix using a factorization:

    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This representation is ameanable to gradient-based learning.

    Input shape:
        (batch_size, dim_x, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim_y, dim_y)
        where dim_y is the dimension of the output state
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        num_eig: int. Number of eigenvectors used to represent 
                 the density matrix
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            dim_y: int = 2,
            num_eig: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y
        if num_eig < 1:
            num_eig = dim_x * dim_y
        self.num_eig = num_eig

    def build(self, input_shape):
        if input_shape[1] != self.dim_x or input_shape[2] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x}, {self.dim_x})')
        self.eig_vec = self.add_weight(
            "eig_vec",
            shape=(self.dim_x * self.dim_y, self.num_eig),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
                          tf.linalg.diag(tf.sqrt(eig_val)))
        rho = tf.matmul(
            rho_h, 
            tf.transpose(rho_h, conjugate=True))
        rho = tf.reshape(
            rho, 
            (self.dim_x, self.dim_y, self.dim_x, self.dim_y))
        rho_res = tf.einsum(
            '...ik, klmn, ...mo -> ...ilon',
            inputs, rho, inputs,
            optimize='optimal')  # shape (b, nx, ny, ny, nx)
        trace_val = tf.einsum('...ijij->...', rho_res, optimize='optimal') # shape (b)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        trace_val = tf.expand_dims(trace_val, axis=-1)
        rho_res = rho_res / trace_val
        rho_y = tf.einsum('...ijik->...jk', rho_res, optimize='optimal') # shape (b, ny, ny)
        return rho_y

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_y, dim_x, dim_y)
        """
        if (len(rho.shape.as_list()) != 4 or
                rho.shape[0] != self.dim_x or
                rho.shape[2] != self.dim_x or
                rho.shape[1] != self.dim_y or
                rho.shape[3] != self.dim_y):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_y},'
                f' {self.dim_x}, {self.dim_y})')
        if not self.built:
            self.build((None, self.dim_x, self.dim_x))
        rho_prime = tf.reshape(
            rho, 
            (self.dim_x * self.dim_y, self.dim_x * self.dim_y,))
        e, v = tf.linalg.eigh(rho_prime)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y, self.dim_y)

class QMeasureDensity(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation.

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input  state
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            **kwargs
    ):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.rho = self.add_weight(
            "rho",
            shape=(self.dim_x, self.dim_x),
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
            '...ik, km, ...mi -> ...',
            oper, self.rho, oper,
            optimize='optimal')  # shape (b, nx, ny, nx, ny)
        return rho_res

    def compute_output_shape(self, input_shape):
        return (1,)

class QMeasureDensityEig(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation.
    Represents the density matrix using a factorization:
    
    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is ameanable to gradient-based learning

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            num_eig: int =0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        if num_eig < 1:
            num_eig = dim_x
        self.num_eig = num_eig

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        self.eig_vec = self.add_weight(
            "eig_vec",
            shape=(self.dim_x, self.num_eig),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
                          tf.linalg.diag(tf.sqrt(eig_val)))
        rho_h = tf.matmul(tf.math.conj(inputs), rho_h)
        rho_res = tf.einsum(
            '...i, ...i -> ...',
            rho_h, tf.math.conj(rho_h), 
            optimize='optimal') # shape (b,)
        return rho_res

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_x)
        Returns:
            e: list of eigenvalues in non-decreasing order
        """
        if (len(rho.shape.as_list()) != 2 or
                rho.shape[0] != self.dim_x or
                rho.shape[1] != self.dim_x):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_x})')
        if not self.built:
            self.build((None, self.dim_x))        
        e, v = tf.linalg.eigh(rho)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_eig ": self.num_eig
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)

def complex_initializer(base_initializer):
    """
    Complex Initializer to use in ComplexQMeasureDensityEig
    taken from https://github.com/tensorflow/tensorflow/issues/17097
    """
    f = base_initializer()

    def initializer(*args, dtype=tf.complex64, **kwargs):
        real = f(*args, **kwargs)
        imag = f(*args, **kwargs)
        return tf.complex(real, imag)

    return initializer

class ComplexQMeasureDensity(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation with complex values.
    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input  state
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            **kwargs
    ):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        with tf.device('cpu:0'):
            self.rho = self.add_weight(
                "rho",
                shape=(self.dim_x, self.dim_x),
                dtype=tf.complex64,
                initializer=complex_initializer(tf.keras.initializers.Zeros),
                trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        rho_res = tf.einsum(
            '...k, km, ...m -> ...',
            tf.math.conj(inputs), self.rho, inputs,
            optimize='optimal')  # shape (b,)
        return rho_res

    def compute_output_shape(self, input_shape):
        return (1,)

class ComplexQMeasureDensityEig(tf.keras.layers.Layer):
    """Quantum measurement layer for density estimation with complex terms.
    Represents the density matrix using a factorization:

    `dm = tf.matmul(V, tf.transpose(V, conjugate=True))`

    This rerpesentation is ameanable to gradient-based learning

    Input shape:
        (batch_size, dim_x)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, 1)
    Arguments:
        dim_x: int. the dimension of the input state
        num_eig: Number of eigenvectors used to represent the density matrix
    """

    @typechecked
    def __init__(
            self,
            dim_x: int,
            num_eig: int =0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        if num_eig < 1:
            num_eig = dim_x
        self.num_eig = num_eig

    def build(self, input_shape):
        if (not input_shape[1] is None) and input_shape[1] != self.dim_x:
            raise ValueError(
                f'Input dimension must be (batch_size, {self.dim_x})')
        with tf.device('cpu:0'):
            self.eig_vec = self.add_weight(
                "eig_vec",
                shape=(self.dim_x, self.num_eig),
                dtype=tf.complex64,
                initializer=complex_initializer(tf.random_normal_initializer),
                trainable=True)
        self.eig_val = self.add_weight(
            "eig_val",
            shape=(self.num_eig,),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True)
        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.complex64)
        norms = tf.expand_dims(tf.linalg.norm(self.eig_vec, axis=0), axis=0)
        eig_vec = self.eig_vec / norms
        eig_val = tf.keras.activations.relu(self.eig_val)
        eig_val = eig_val / tf.reduce_sum(eig_val)
        rho_h = tf.matmul(eig_vec,
                          tf.cast(tf.linalg.diag(tf.sqrt(eig_val)), 
                                  dtype=tf.complex64))
        rho_h = tf.matmul(tf.math.conj(inputs), rho_h)
        rho_res = tf.einsum(
            '...i, ...i -> ...',
            rho_h, tf.math.conj(rho_h), 
            optimize='optimal') # shape (b,)
        rho_res = tf.cast(rho_res, tf.float32)
        return rho_res

    def set_rho(self, rho):
        """
        Sets the value of self.rho_h using an eigendecomposition.

        Arguments:
            rho: a tensor of shape (dim_x, dim_x)
        Returns:
            e: list of eigenvalues in non-decreasing order
        """
        if (len(rho.shape.as_list()) != 2 or
                rho.shape[0] != self.dim_x or
                rho.shape[1] != self.dim_x):
            raise ValueError(
                f'rho shape must be ({self.dim_x}, {self.dim_x})')
        if not self.built:
            self.build((None, self.dim_x))        
        e, v = tf.linalg.eigh(rho)
        self.eig_vec.assign(v[:, -self.num_eig:])
        self.eig_val.assign(e[-self.num_eig:])
        return e

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_eig ": self.num_eig
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (1,)

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

class ComplexDensityMatrix2Dist(tf.keras.layers.Layer):
    """Extracts a probability distribution from a complex density matrix.

    Input shape:
        A tensor with shape (batch_size, n, n)
    Output shape:
        (batch_size, n)
    Arguments:
    """

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
        cp = tf.cast(cp, tf.float32)
        return cp

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[1])


class DensityMatrixRegression(tf.keras.layers.Layer):
    """
    Calculates the expected value and variance of a measure on a
    density matrix. The measure associates evenly distributed values
    between 0 and 1 to the different n basis states.

    Input shape:
        A tensor with shape (batch_size, n, n)
    Output shape:
        (batch_size, n, 2)
    Arguments:
    """

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
        self.vals = tf.constant(tf.linspace(0., 1., input_shape[1]),
                                dtype=tf.float32)
        self.vals2 = self.vals ** 2
        self.built = True

    def call(self, inputs):
        if len(inputs.shape) != 3 or inputs.shape[1] != inputs.shape[2]:
            raise ValueError('A `DensityMatrix2Dist` layer should be '
                             'called with a tensor of shape '
                             '(batch_size, n, n)')
        mean = tf.einsum('...ii,i->...', inputs, 
                         self.vals, optimize='optimal')
        mean2 = tf.einsum('...ii,i->...', inputs, 
                          self.vals2, optimize='optimal')
        var = mean2 - mean ** 2
        return tf.stack([mean, var], axis = -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[1], 2)

class ComplexDensityMatrixRegression(tf.keras.layers.Layer):
    """
    Calculates the expected value and variance of a measure on a 
    density matrix. The measure associates evenly distributed values 
    between 0 and 1 to the different n basis states.

    Input shape:
        A tensor with shape (batch_size, n, n)
    Output shape:
        (batch_size, n, 2)
    Arguments:
    """

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
        self.vals = tf.cast(tf.constant(tf.linspace(0., 1., input_shape[1]), 
                            dtype=tf.float32), tf.complex64)
        self.vals2 = self.vals ** 2
        self.built = True

    def call(self, inputs):
        if len(inputs.shape) != 3 or inputs.shape[1] != inputs.shape[2]:
            raise ValueError('A `DensityMatrix2Dist` layer should be '
                             'called with a tensor of shape '
                             '(batch_size, n, n)')
        mean = tf.einsum('...ii,i->...', inputs, self.vals, optimize='optimal')
        mean2 = tf.einsum('...ii,i->...', inputs, self.vals2,
            optimize='optimal')
        mean = tf.cast(mean, tf.float32)
        mean2 = tf.cast(mean2, tf.float32)
        var = mean2 - mean ** 2
        return tf.stack([mean, var], axis = -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[1], 2)
