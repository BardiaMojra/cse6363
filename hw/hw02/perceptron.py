
import pdb
import numpy as np

data = [( 1,  1,  1,  1,  1),
( 1,  1, -1,  1,  1),
( 1, -1,  1,  1,  1),
( 1, -1, -1,  1,  1),
(-1,  1,  1,  1,  1),
(-1,  1, -1,  1,  1),
(-1, -1,  1,  1,  1),
(-1, -1, -1,  1, -1)]

class perceptron:
  def __init__(X, Y, lr):
    self.lr = lr
    self.X = X
    self.Y = Y
    self.m = len(X[0]) # input feature set size
    self.n = len(X) # dataset size

    return

def get_data(data):
  X = list()
  Y = list()
  for i in range(len(data)):
    X.append(np.asarray(data[i][:-1]))
    Y.append(np.asarray(data[i][-1]))
  X = np.asarray(X)
  Y = np.asarray(Y)
  Y = np.expand_dims(Y,axis=1)

  pdb.set_trace()
  return X, Y



if __name__ == '__main__':

  X, Y = get_data(data)

  ptron = perceptron(X,Y,lr=.01)
