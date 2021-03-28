import numpy as np
import pdb


class dlnet:
  def __init__(self, x, y):
    self.X = np.concatenate((x, np.zeros((8,1))), axis = 1)
    self.Y =y
    self.Yest=np.zeros((1,self.Y.shape[1]))
    self.dims = [5, 5, 1]
    self.param = {}
    self.ch = {}
    self.loss = []
    self.lr=0.03
    self.sam = self.Y.shape[1]


  def nInit(self):
    np.random.seed(1)
    self.W1 = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
    self.W2 = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
    return

  def nloss(self,Yest): # batch normalized cross entropy loss
    loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yest).T) - np.dot(1-self.Y, np.log(1-Yest).T))
    return loss

  def forward(self):
        self.Z1 = self.X.dot(self.W1.T)
        self.A1 = Sigmoid(self.Z1)

        self.Z2 = self.A1.dot(self.W2.T)

        self.A2 = Sigmoid(self.Z2)
        self.Yest = hardlim(self.Z2)
        loss = self.nloss(self.A2)
        return self.Yest, loss

  def backward(self):
    # get gradient loss of Yest
    dLoss_Yest = - (np.divide(self.Y, self.Yest ) - np.divide(1 - self.Y, 1 - self.Yest))

    dLZ2 = dLoss_Yest * dSigmoid(self.Z2)
    dLA1 = np.dot(dLZ2, self.W2) #

    # dW2 1x5
    dLoss_W2 = 1./self.A1.shape[1] * np.dot(dLZ2.T, self.A1)

    dLZ1 = dLA1 * dSigmoid(self.Z1)
    dLA0 = np.dot(self.W1,dLZ1.T)
    dLW1 = 1./self.X.shape[1] * np.dot(self.X.T, dLZ1)

    self.W1 = self.W1 - self.lr * dLW1
    self.W2 = self.W2 - self.lr * dLoss_W2


  def BGD(self,X, Y, iter = 1000):
    np.random.seed(1)
    self.nInit()
    for i in range(0, iter):
      Yest, loss=self.forward()
      self.backward()
      if i % 50 == 0:
        print ("Cost after iteration {}".format(i),": {}".format(loss))
        self.loss.append(loss)
    return


def Sigmoid(Z):
  return 1/(1+np.exp(-Z))

def Relu(Z):
  return np.maximum(0,Z)

def dRelu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

def dSigmoid(Z):
  s = 1/(1+np.exp(-Z))
  dZ = s * (1-s)
  return dZ

def hardlim(vec):
  res = np.zeros(vec.shape)
  for i in range(len(vec)):
    if vec[i,:] > .5: res[i,:] = 1
  else: res[i,:] = 0
  return res

def get_data(data, Ymax,  Ymin):
  X = list()
  Y = list()
  for i in range(len(data)):
    X.append(np.asarray(data[i][:-1]))
    Y.append(np.asarray(data[i][-1]))
  X = np.asarray(X)
  Y = [Ymax if i > 0 else Ymin for i in Y]
  Y = np.asarray(Y)
  Y = np.expand_dims(Y,axis=1)
  XY = np.concatenate((X, Y), axis=1)
  print('dataset: ')
  print(XY)
  print('\n\n')
  return X, Y
dataOR = [( 1,  1,  1,  1,  1),
          ( 1,  1, -1,  1,  1),
          ( 1, -1,  1,  1,  1),
          ( 1, -1, -1,  1,  1),
          (-1,  1,  1,  1,  1),
          (-1,  1, -1,  1,  1),
          (-1, -1,  1,  1,  1),
          (-1, -1, -1,  1, -1)]

dataXOR = [( 1,  1,  1,  1, -1),
           ( 1,  1, -1,  1,  1),
           ( 1, -1,  1,  1,  1),
           ( 1, -1, -1,  1,  1),
           (-1,  1,  1,  1,  1),
           (-1,  1, -1,  1,  1),
           (-1, -1,  1,  1,  1),
           (-1, -1, -1,  1, -1)]


if __name__ == '__main__':

  X, Y = get_data(dataOR, Ymax=1, Ymin=0) # rescale output to [0,1]
  nn = dlnet(X, Y)
  nn.BGD(X, Y, iter = 15000)
