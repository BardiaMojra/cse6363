
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
from pdb import *
from pprint import pprint as pp
import copy

trainXY = [ [170, 57, 32, 'W'],
[190, 95, 28, 'M'],
[150, 45, 35, 'W'],
[168, 65, 29, 'M'],
[175, 78, 26, 'M'],
[185, 90, 32, 'M'],
[171, 65, 28, 'W'],
[155, 48, 31, 'W'],
[165, 60, 27, 'W'],
[182, 80, 30, 'M'],
[175, 69, 28, 'W'],
[178, 80, 27, 'M'],
[160, 50, 31, 'W'],
[170, 72, 30, 'M'], ]

testXY = [ [155,40,35, 'W'],
 [170,70,32, 'M'],
 [175,70,35, 'W'],
 [180,90,20, 'M'], ]






def sigmoid(x):
  return 1/(1+np.exp(-x))

def get_cost(X, y, theta):
  n = len(y)
  h = sigmoid(X@theta)
  eps = 1e-5
  cost = (1/n)* (((-y).T @ np.log(h+eps))-((1-y).T @ np.log(1-h+eps)))
  return cost

def grad_desc( X, y, params, lr, iters):
  n = len(y)
  cost_hist = np.zeros((iters, 1))
  for i in range(iters):
    params = params - (lr/n) * (X.T @ (sigmoid(X@params)-y))
    cost_hist[i] = get_cost(X, y, params)
  return cost_hist, params

def predict(X, params):
  return np.round(sigmoid(X@params))

def get_data(dataXY):
  dataXY = np.array(dataXY)
  X = np.array(dataXY[:,:-1], dtype='float64')
  y = dataXY[:,-1]
  for i in range(len(y)):
    if y[i] == 'M': y[i] = 0
    if y[i] == 'W': y[i] = 1
  return X, np.array(y[:,np.newaxis], dtype='float64')

def get_acc(yest, y, window=50):
  #y = y[:-window]
  #yest = yest[:-window]
  #set_trace()
  score = float(sum(yest == y))/ float(len(y))
  return round((score*100), 3)


if __name__ == '__main__':

  pp('Training set:')
  pp(trainXY)
  print()

  pp('Test set:')
  pp(testXY)
  print()

  trainX, trainY = get_data(trainXY)
  testX, testY = get_data(testXY)

  sns.set_style('white')

  m = trainY.shape[0]
  trainX = np.hstack((np.ones((m,1)),trainX))
  n = np.size(trainX,1)
  params = np.zeros((n,1))

  iters = 1000
  lr = 0.001

  init_cost = get_cost(trainX, trainY, params)


  print('Initial cost: {}'.format(init_cost))
  print()

  cost_hist, params = grad_desc(trainX, trainY, params, lr, iters)


  print('Final cost: {}'.format(cost_hist[-1]))
  print()
  print('Final params: ')
  print(params)
  print()

  plt.figure()
  sns.set_style('white')
  plt.plot(range(len(cost_hist)), cost_hist, 'r')
  plt.title("Cost Gradient Descent")
  plt.xlabel("Iterations")
  plt.ylabel("Cost")
  plt.savefig('./p02_log_reg_fig01_Cost_vs_GD.png', dpi=200)
  #plt.show()


  yest = predict(trainX, params)
  score = get_acc(yest, trainY)
  print('train accuracy: {}'.format(score))

  m = testX.shape[0]
  b = np.ones((m,1))
  testX = copy.deepcopy(np.concatenate((b,testX), axis=1))
  yest = predict(testX, params)
  score = get_acc(yest, testY)
  print('test accuracy: {}'.format(score))

  print('---->>> end of process.')





# EOF
