
import pdb
import numpy as np
import random

data = [( +1,  1,  1,  1,  1),
( 1,  1, -1,  1,  1),
( 1, -1,  1,  1,  1),
( 1, -1, -1,  1,  1),
(-1,  1,  1,  1,  1),
(-1,  1, -1,  1,  1),
(-1, -1,  1,  1,  1),
(-1, -1, -1,  1, -1)]

NBUG = 1
class perceptron:
  def __init__(self, lr, iters):
    self.lr = lr
    self.iters = iters
    self.af = self.sign
    self.W = None
    self.b = None
    return

  def sign(self, val):
    return np.where(val>=0, 1, -1)

  def train(self, X, Y):
    self.nSamples, self.mFeatures = X.shape
    self.W = np.zeros(self.mFeatures)
    self.b = 0

    XY = np.concatenate((X, Y), axis=1)
    XY_tmp = np.ndarray.copy(XY)
    for _ in range(self.iters):
      i = random.randint(0, len(XY_tmp)-1)
      Xdatum = XY_tmp[i][:-1]
      Ydatum = XY_tmp[i][-1]
      XY_tmp = np.delete(XY_tmp, i, axis=0)

      #pdb.set_trace()
      linOutput = np.dot(Xdatum, self.W) + self.b
      y_ = self.af(linOutput)
      # apply the Perceptron rule
      update = self.lr * (Ydatum - y_)
      self.W += update * Xdatum
      self.b += update
      if len(XY_tmp)==0: XY_tmp = np.ndarray.copy(XY)
      if NBUG==1:
        print("iter:", _, " |X:", Xdatum, " |Y:", Ydatum, " |y_:", y_)

    return

  def predict(self, X):
    dotprod = np.dot(X, self.W) + self.b
    return self.sign(dotprod)


def get_data(data):
  X = list()
  Y = list()
  for i in range(len(data)):
    X.append(np.asarray(data[i][:-1]))
    Y.append(np.asarray(data[i][-1]))
  X = np.asarray(X)
  Y = [1 if i > 0 else 0 for i in Y ]
  Y = np.asarray(Y)
  Y = np.expand_dims(Y,axis=1)
  return X, Y



def get_acc(self, groundtruth, prediction):
  correct = 0
  for i in range(len(groundtruth)):
    if groundtruth[i] == prediction[i]:
      correct += 1
  return correct / float(len(groundtruth)) * 100.0



if __name__ == '__main__':

  X, Y = get_data(data)
  p = perceptron(.01, 100)
  p.train(X,Y)
  y_est = p.predict(X[3])
