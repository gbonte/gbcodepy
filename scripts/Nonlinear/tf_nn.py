# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

# Automatic differentiation example
# Based on https://tensorflow.rstudio.com/guides/tensorflow/autodiff

import tensorflow as tf
import numpy as np



def g(z):
    return 1 / (1 + tf.exp(-z))

def gp(z):
    return tf.exp(-z) / (1 + tf.exp(-z))**2

# Create a constant with a random normal value
x = tf.constant(np.random.randn(), dtype=tf.float32)

# Initialize weights as TensorFlow Variables with random normal values
w11_1 = tf.Variable(np.random.randn(), dtype=tf.float32)
w12_1 = tf.Variable(np.random.randn(), dtype=tf.float32)

w11_2 = tf.Variable(np.random.randn(), dtype=tf.float32)
w21_2 = tf.Variable(np.random.randn(), dtype=tf.float32)

# Use GradientTape to record operations for automatic differentiation
with tf.GradientTape() as tape:
    a1_1 = w11_1 * x
    z1 = g(a1_1)
    
    a2_1 = w12_1 * x
    z2 = g(a2_1)  # 1/(1+exp(-a2_1))
    
    a1_2 = w11_2 * z1 + w21_2 * z2
    yhat = g(a1_2)  # 1/(1+exp(-a1_2))

# Compute gradients of yhat with respect to the weights
gradients = tape.gradient(yhat, [w11_1, w12_1, w11_2, w21_2])

# Compute analytical gradients
g11_1 = gp(a1_2) * w11_2 * gp(a1_1) * x   # dyhat/dw11_1
g12_1 = gp(a1_2) * w21_2 * gp(a2_1) * x   # dyhat/dw12_1
g11_2 = gp(a1_2) * z1                     # dyhat/dw11_2
g21_2 = gp(a1_2) * z2                     # dyhat/dw21_2

print("TF=", float(gradients[0].numpy()), " Analytical=", float(g11_1.numpy()))
print("TF=", float(gradients[1].numpy()), " Analytical=", float(g12_1.numpy()))
print("TF=", float(gradients[2].numpy()), " Analytical=", float(g11_2.numpy()))
print("TF=", float(gradients[3].numpy()), " Analytical=", float(g21_2.numpy()))
