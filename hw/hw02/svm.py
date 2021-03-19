
import pdb
import numpy as np
import matplotlib.pyplot as plt
from random import random

plt.style.use('ggplot')


D = [( 3,  4, -1),
     ( 2,  3, -1),
     ( 2,  1,  1),
     ( 1,  2,  1),
     ( 1,  3,  1),
     ( 4,  4, -1)]

class svm:
  def __init__(self, lr=0.01, lp, iters=1000, prt=False, log=False):
    self.lr = lr # learning rate
    self.lp = lp # lambda parameter
    self.iters = iters
    self.prt = prt
    self.w = None
    self.b = None
    if log==True:
      self.xHist = list()
      self.yHist = list()
      self.yEstHist =list()
      self.wHist = list()
      self.bHist = list()
      self.wUpdHist = list()
      self.bUpdHist = list()
      # add logs
    return

  def fit(self, X, Y):
    nSamples, mfeatures = X.shape
    self.w = np.zeros(mfeatures)
    self.b = 0

    XY = np.concatenate((X, Y), axis=1)

    XY_tmp = np.ndarray.copy(XY)
    for j in range(self.iters):
      a = random.randint(0, len(XY_tmp)-1)
      xDatum = XY_tmp[a][:-1]
      yDatum = XY_tmp[a][-1]
      XY_tmp = np.delete(XY_tmp, a, axis=0)
      #pdb.set_trace()
      yEst = yDatum * (np.dot(xDatum, self.w) - self.b) >= 1
      if yEst: # binary classification only
        wUpdate = self.lr * (2 * self.lp * self.w)
        bUpdate = 0.0
      else:
        wUpdate = self.lr * (2 * self.lp * self.w - np.dot(xDatum, yDatum))
        bUpdate = self.lr * yDatum
      self.w -= wUpdate
      self.b -= bUpdate
      if len(XY_tmp)==0: XY_tmp = np.ndarray.copy(XY) # reload!!
      if self.prt:
        print('iter:{:3.d}'.format(j))
      if self.log: # data logger!!
        self.xHist.append(xDatum)
        self.yHist.append(yDatum)
        self.yEstHist.append(yEst)
        self.wUpdHist.append(wUpdate)
        self.bUpdHist.append(bUpdate)
        self.wHist.append(self.w)
        self.bHist.append(self.b)
    return

  def predict(self, X):
    linOutput = np.dot(X, self.w) - self.b
    return np.sign(linOutput)

  def getHyperPlaneVal(self, x, w, b, offset):
    res = (-w[0] * x + b + offset)/w[1]
    return res

  def visualize(self, X, Y):
    fig = plt.figure()
    plott = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], marker='x', color='red', label='input')

    x01 = np.amin(X[:,0])
    x02 = np.amax(X[:,0])



    return


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
