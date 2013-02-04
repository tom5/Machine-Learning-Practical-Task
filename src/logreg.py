import numpy as np
from util import *

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def log_sigmoid(z):
  # n.b. -np.logaddexp(0,-z) calculates -log(1+exp(-z)) 
  # in log space without exponentiation to avoid overflow.
  # Try help(numpy.logaddexp) in the interpreter for
  # more information.
  return -np.logaddexp(0,-z)

def log_sigmoid_complement(z):
  return -np.logaddexp(0,z)

# This function should calculate the negative conditional 
# log probability of the data x given weights w and 
# observed response variables y.
def objective(x, y, w):
  total = 0
  for i in range(0, len(x)):
    total += y[i] * log_sigmoid(np.dot(x[i], w))
    total += (1 - y[i]) * log_sigmoid_complement(np.dot(x[i], w))
  return -total

# This is the log Gaussian prior 
def log_prior(w, alpha):
  return np.dot(w, w) / (2*alpha)

# This function should calculate the gradient of the negative 
# conditional log probability of the data x given weights w
# and observed response variables y.
def grad(x, y, w):
  grad_vector = np.zeros(len(x[0]))
  for i in range(0, len(x)):
    weight = (sigmoid(np.dot(x[i], w)) - y[i])
    for j in range(0, len(x[i])):
      grad_vector[j] += weight * x[i][j]
  return grad_vector

# This is the derivative of the Gaussian prior
def prior_grad(w, alpha):
  grad_vector = np.zeros(len(w))
  for i in range(len(w)):
    grad_vector[i] = w[i] / (alpha * alpha)
  return grad_vector
