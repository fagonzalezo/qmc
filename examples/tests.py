"""
Tests
"""
import sys
sys.path.insert(0, "../")

import tensorflow as tf
import numpy as np
from qmc.tf import layers
from qmc.tf import models


data_x = tf.constant([[0., 0.],
                   [0., 1,],
                   [1., 0,],
                   [1., 1,]
                  ])
data_y = tf.constant([[0], [1], [1], [0]])
fm_x = layers.QFeatureMapRFF(input_dim=2, dim=10, gamma=4, random_state=10)
qmd = models.QMDensity(fm_x)
out = qmd(data_x)
print(out)
qmd.compile()
qmd.fit(data_x, data_y, epochs=1)
out = qmd(data_x)
print(out)


#fm_x = layers.QFeatureMapSmp(dim=2, beta=10)
fm_x = layers.QFeatureMapRFF(input_dim=2, dim=10, gamma=4, random_state=10)
fm_y = layers.QFeatureMapOneHot(num_classes=2)
#fm_y = lambda x: tf.squeeze(tf.keras.backend.one_hot(x, 2))
qmc = models.QMClassifier(fm_x, fm_y, dim_y=2)
out = qmc(data_x)
print(out)
qmc.compile()
qmc.fit(data_x, data_y, epochs=1)
out = qmc(data_x)
print(out)

qmc1 = models.QMClassifierSGD(2, 10, 2, gamma=4, random_state=10)
out = qmc1(data_x)
qmc1.set_rho(qmc.weights[2])
out1 = qmc1(data_x)
print(out1)

qmc2 = models.QMClassifierSGD(2, 10, 2, gamma=4, random_state=10)
out = qmc2(data_x)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
qmc2.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
train_y = tf.reshape (tf.keras.backend.one_hot(data_y, 2), (4,2))
qmc2.fit(data_x, train_y, epochs=20)
out2 = qmc2(data_x)
print(out2)


out = qmc.call_train(data_x, data_y)
print(out)
qmc.train_step((data_x, data_y))

inputs = tf.ones((2,3))
qfm = layers.QFeatureMapRFF()
out1 = qfm(inputs)
print(out1)

inputs = tf.ones((2,3))
qfm = layers.QFeatureMapSmp(dim=2, beta=10)
out1 = qfm(inputs)
print(out1)
qmeas = layers.QMeasureClassifEig(dim_x=8, dim_y=3, num_eig=0)
out2 = qmeas(out1)
print(out2)
dmr = layers.DensityMatrixRegression()
out3 = dmr(out2)
print(out3)


out3 = layers.CrossProduct()([out1, out2])
print(out3)
out4 = layers.DensityMatrix2Dist()(out2)
print(out4)
