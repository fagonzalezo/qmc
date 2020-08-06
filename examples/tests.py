"""
Tests
"""
import tensorflow as tf
import numpy as np
from qmc.tf import layers
from qmc.tf import models

inputs = tf.ones((2,3))
qfm = layers.QFeatureMapRFF()
out1 = qfm(inputs)
print(out1)

inputs = tf.ones((2,3))
qfm = layers.QFeatureMapSmp(dim=2, beta=10)
out1 = qfm(inputs)
qmeas = layers.QMeasurement(dim_y=8)
print(out1)
out2 = qmeas(out1)
print(out2)
out3 = layers.CrossProduct()([out1, out2])
print(out3)
out4 = layers.DensityMatrix2Dist()(out2)
print(out4)

inputs = tf.ones((2,3))
data_x = tf.constant([[0., 0.],
                   [0., 1,],
                   [1., 0,],
                   [1., 1,]
                  ])
data_y = tf.constant([[0], [1], [1], [0]])
#fm_x = layers.QFeatureMapSmp(dim=2, beta=10)
fm_x = layers.QFeatureMapRFF(dim=10, gamma=4)
fm_y = layers.QFeatureMapOneHot(num_classes=2)
#fm_y = lambda x: tf.squeeze(tf.keras.backend.one_hot(x, 2))
qmc = models.QMClassifier(fm_x, fm_y, dim_y=2)
out = qmc(data_x)
print(out)
qmc.compile()
qmc.fit(data_x, data_y, epochs=1)
out = qmc(data_x)
print(out)

out = qmc.call_train(data_x, data_y)
print(out)
qmc.train_step((data_x, data_y))


