
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt

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

class nn_2layer:
  def __init__(self, X, Y, lr, iters, lossFunc='L2', prt=True, recTrainHist=True):
    self.X = X
    self.nSamples, self.mFeatures = X.shape
    self.Y = Y
    self.Yest = np.zeros((1,1)) # network's current estimate
    self.numLayers = 2
    iSize = self.mFeatures
    hSize = 4 # nodes (units) for the hidden layer
    oSize = 1
    self.nnDims = [iSize, hSize, oSize]
    self.lr = lr
    self.iters = iters
    self.printRate = iters / 10
    self.lossFunc = lossFunc
    #self.pIC = paramIC # parameter initial conditions - normal or uniform distr.
    self.loss = 0 # training loss
    #self.param = dict() # a dictionary to hold all parameters
    self.grad = dict() # a dict to hold all gradients (for SGD at BProp)
    #self.gCache = dict() # will need this for chain rule at BProp
    # Print and data logging
    self.prt = prt
    self.log.on = recTrainHist # this is the Enable flag for data logging

    # uniform distribution
    # first layer
    self.W1 = np.random.uniform(low=-.1, high=.1, \
      size=(self.nnDims[1], self.nnDims[0])) / np.sqrt(self.nnDims[0]) # we normalize W vectors by dividing by the sqrt of size of the previous layer
    self.B1 = np.random.uniform(low=-.1, high=.1, size=(self.nnDims[1], 1))
    self.Z1 = np.zeros((hSize, oSize))
    self.Y1 = np.zeros((hSize, oSize))
    # second layer
    self.W2 = np.random.uniform(low=-.1, high=.1, \
      size=(self.nnDims[2], self.nnDims[1])) / np.sqrt(self.nnDims[1]) # we normalize W vectors by dividing by the sqrt of size of the previous layer
    self.B2 = np.random.uniform(low=-.1, high=.1, size=(self.nnDims[2], 1))
    self.Z2 = np.zeros((oSize, 1))
    self.Yest = np.zeros((oSize, 1))
    if self.prt:
      print('Initialize model parameters:')
      print('W1:', self.W1)
      print('B1:', self.B1)
      print('W2:', self.W2)
      print('B2:', self.B2)
      print('\n\n')
    #pdb.set_trace()
    if self.log.on:
      self.log.W1 = list()
      self.log.B1 = list()
      self.log.Z1 = list()
      self.log.Y1 = list()
      self.log.W2 = list()
      self.log.B2 = list()
      self.log.Z2 = list()
      self.log.Yest = list()
      self.log.iters = list()
      self.log.W1update = list()
      self.log.B1update = list()
      self.log.W2update = list()
      self.log.B2update = list()
      self.log.accHist = list()
    return


  def sigmoid(self, var):
    res = 1/(1 + np.exp(-var))
    #pdb.set_trace()
    return res

  def dSigmoid(self, var):
    res = self.sigmoid(var) * (1.0 - self.sigmoid(var))
    return res

  # forward pass is basically an estimator
  def feedforward(self, iteration):
    # 1st layer
    self.Z1 = self.W1.dot(self.X) + self.B1
    self.Y1 = self.sigmoid(self.Z1)
    # 2nd layer
    self.Z2 = self.W2.dot(self.Y1) + self.B2
    self.Yest = self.sigmoid(self.Z2)
    # calc loss
    self.loss = self.lossFunc(self.Y, self.Yest)
    # log data
    if self.log.on:
      self.log.Z1.append(self.Z1)
      self.log.Y1.append(self.Y1)
      self.log.Z2.append(self.Z2)
      self.log.Yest.append(self.Yest)
      self.log.loss.append(self.loss)
      self.log.iters.append(iteration)
    return

  def L1(self, Yact, Yest):
    losses = abs(Yact-Yest)
    loss = np.sum(losses)
    return loss

  def L2(self, Yact, Yest):
    losses = (Yact-Yest) ** 2
    loss = np.sum(losses)
    return loss

  def cEntropy(self, Yact, Yest): # not tested
    loss = (1.0 / self.nSamples) * (- np.dot(Yact, np.log(Yest).T) - \
    np.dot(1-Yact, np.log(1-Yest).T))
    return loss

  def dLossYest(self, Yact, Yest):
    dLossYest = - ((Yact/Yest) - ((1-Yact)/(1-Yest)))
    return dLossYest

  def backprop(self):
    '''
      Backpropagation is done by performing partial derivative of the entire
      network (the output loss - end of feedforward) with respect to the network
      parameters, W1, B1, W2, B2. Once output sensitivity (rate of change) is
      determined with respect to each parameter, we update them accordingly.
    '''
    # start with the output
    dLossYest = - (np.divide(self.Y, self.Yest) - np.divide(1-self.Y, 1-self.Yest))
    # 2nd layer
    dLossZ2 = dLossYest * self.dSigmoid(self.Z2)
    dLossW2 = 1.0 / self.Y1.shape[1] * np.dot(dLossZ2, self.Y1.T)
    dLossB2 = 1.0 / self.Y1.shape[1] * np.dot(dLossZ2, np.ones([dLossZ2.shape[1], 1]))
    dLossY1 = np.dot(self.W2.T, dLossZ2)
    # 1st layer
    dLossZ1 = dLossY1 * self.dSigmoid(self.Z1)
    dLossW1 = 1.0 / self.X.shape[1] * np.dot(dLossZ1, self.X.T)
    dLossB1 = 1.0 / self.X.shape[1] * np.dot(dLossZ1, np.ones([dLossZ1.shape[1], 1]))
    dLossY0 = np.dot(self.W1.T, dLossZ1)
    # produce updates for each parameter
    W1update = - (self.lr * dLossW1)
    B1update = - (self.lr * dLossB1)
    W2update = - (self.lr * dLossW2)
    B2update = - (self.lr * dLossB2)
    # update network parameters
    self.W1 += W1update
    self.B1 += B1update
    self.W2 += W2update
    self.B2 += B2update
    # log updates and updated parameters
    if self.log.on:
      self.log.W1update.append(W1update)
      self.log.B1update.append(B1update)
      self.log.W2update.append(W2update)
      self.log.B2update.append(B2update)
      self.log.W1.append(self.W1)
      self.log.B1.append(self.B1)
      self.log.W2.append(self.W2)
      self.log.B2.append(self.B2)
    return

  def train(self):
    iters = self.iters
    print('In training...')
    # train with Gradient Descent - the dataset is too small for SGD or BGD
    for i in range(0, iters):
      self.feedforward(i)
      self.backprop()

      if ((i % self.printRate == 0) & (self.prt)):
        print('Forward pass:')
        print('Z1:', self.Z1)
        print('Y1:', self.Y1)
        print('Z2:', self.Z2)
        print('Yest (Yest):', self.Yest)
        print('Loss:', self.loss)
        print('Backprop:')
        print('W1:', self.W1)
        print('B1:', self.B1)
        print('W2:', self.W2)
        print('B2:', self.B2)
        print('\n\n')
    return


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




if __name__ == '__main__':
  print('-->> Training and testing with OR')
  X, Y = get_data(dataOR, Ymax=1, Ymin=0) # rescale output to [0,1] since we're using Sigmoid activation function

  nnOR = nn_2layer(X, Y, lr=.01, iters=1000)
  nnOR.train()
  #plt.plot(range(len(pOR.accHist)), nnOR.accHist, label='OR Perceptron Taining Acc')
  #plt.show()
  #figOR = plt.figure()
  #ax = figOR.add_subplot(111, projection='3d')



  print("\n\n")
  print('-->> Training and testing with XOR')
  xorX, xorY = get_data(dataXOR, Ymax=1, Ymin=0) # rescale output to [0,1] since we're using Sigmoid activation function
  pXOR = nn_2layer(.01, 1000, True, L2, True)
  pXOR.train(xorX,xorY)
  plt.style.use('ggplot')
  #plt.style.use('fivethirtyeight')

  plt.plot(range(len(pXOR.accHist)), pXOR.accHist, label='XOR Perceptron Taining Acc')

  plt.xlabel('Training iterations')
  plt.ylabel('Training accuracy')
  plt.title('OR vs. XOR training accuracy')
  plt.legend()
  #plt.grid(True)
  #plt.tight_layout()
  plt.show()
