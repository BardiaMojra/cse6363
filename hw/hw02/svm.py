
import pdb
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


D = [( 3,  4, -1),
     ( 2,  3, -1),
     ( 2,  1,  1),
     ( 1,  2,  1),
     ( 1,  3,  1),
     ( 4,  4, -1)]

class svm:
  def __init__(self, lr, lambda_param, iters, prt, log):
    self.lr = lr
    self.lp = lambda_param
    self.iters = iters
    self.prt = prt
    self.w = None
    self.b = None
    if log==True:
      self.xHist = list()
      self.yHist = list()
      self.yEstHist = list()
      self.wHist = list()
      self.bHist = list()
      # add logs
    return

  def fit(self, X, Y):
    nSamples, mfeatures = X.shape
    self.w = np.zeros(mfeatures)
    self.b = 0

    for i in range(self.iters):
      for idx, datum in enumerate(X):
        condition = Y[idx] * (np.dot(datum, self.w) - self.b) >= 1
        if condition==True:

    return

  def predict(self, X):
    linOutput = np.dot(X, self.w) - self.b
    return np.sign(linOutput)


def get_data(data):
  X = list()
  Y = list()
  for i in range(len(data)):
    X.append(np.asarray(data[i][:-1]))
    Y.append(np.asarray(data[i][-1]))
  X = np.asarray(X)
  Y = [1 if i > 0 else -1 for i in Y]
  Y = np.asarray(Y)
  Y = np.expand_dims(Y,axis=1)
  return X, Y

if __name__ == '__main__':

  X, Y = get_data(D)
  for i in range(len(X)):
    if Y[i]>=0:
      plt.scatter(X[i][0], X[i][1], marker='x', color='red')
    else:
      plt.scatter(X[i][0], X[i][1], marker='o', color='blue')
  plt.title('Data Classification')
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.legend()
  plt.show()
