
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg.linalg import eigvals
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


class LDA:
  def __init__(self, nComp=None):
    self.nComp=nComp
    self.LinDiscriminants = None # the components

  def fit(self, X, y):
    nFeat = X.shape[1]  # nSamp x nFeat
    cLabels = np.unique(y)
    nClass = len(cLabels)
    S_t = 0 # temp matrix used for computing scatter
    S_w = 0 # intra-class covariance of the input data
    S_b = 0 # inter-class covariance of the input data


    mu_overall = np.mean(X, axis=0)
    S_w = np.zeros((nFeat, nFeat)) # nFeat X nFeat (3x3)
    S_b = np.zeros((nFeat, nFeat)) # nFeat X nFeat (3x3)

    for c in cLabels:
      X_c = np.where(y==c)[0] # get all data with same label
      X_c = np.array([ X[i] for i in X_c])
      #set_trace()
      mu_c = np.mean(X_c, axis=0) # get their mu
      # make sure the size stays valid
      # (3, 2) x (2, 3) --> (3x3)
      S_w += (X_c - mu_c).T.dot(X_c - mu_c)
      print('S_w:')
      #set_trace()
      nClass = X_c.shape[0]
      mu_diff = (mu_c - mu_overall).reshape(nFeat, 1) # (4x1)
      #set_trace()
      S_b += (mu_diff).dot(mu_diff.T)

    A = np.linalg.inv(S_w).dot(S_b) # get inverse
    eVals, eVecs = np.linalg.eig(A) # get eigen vectors and values
    eVecs = eVecs.T
    idxs = np.argsort(abs(eVals))[::-1] # sort in decreasing order by reversing the indices
    eVals = eVals[idxs]
    eVecs = eVecs[idxs]
    # we only need the first n eVects  (number of dimensions to keep)
    self.LinDiscriminants = eVecs[0:self.nComp]
    return

  def transform(self, X): # project the data in LDA feature space
    return np.dot(X, self.LinDiscriminants.T)

  def plot(self, X_lda):
    if self.nComp == 2:
      if trainY is None:
        plt.scatter(X_lda[:,0],X_lda[:,1])
      else:
        colors = ['b','g','r','c','m','y','k']
        labels = np.unique(trainY)
        for color, label in zip(colors,labels):
          class_data = X_lda[np.flatnonzero(trainY==label)]
          plt.scatter(class_data[:,0], class_data[:,], c=color)
      plt.show()
      plt.savefig('proj01_part03_LDA_fig01.png', dpi=200)
    return

def get_data(dataXY, ):
  dataXY = np.array(dataXY)
  X = np.array(dataXY[:,:-1], dtype='float64')
  y = dataXY[:,-1]
  for i in range(len(y)):
    if y[i] == 'M': y[i] = 0
    if y[i] == 'W': y[i] = 1
  return X, np.array(y[:,np.newaxis])

def get_acc(yest, y, window=50):
  #y = y[:-window]
  #yest = yest[:-window]
  #set_trace()
  score = float(sum(yest == y))/ float(len(y))
  return round((score*100), 3)


if __name__ == '__main__':
  plt.style.use('ggplot')
  print('Training set:')
  pp(trainXY)
  print()

  print('Test set:')
  pp(testXY)
  print()

  trainX, trainY = get_data(trainXY)
  testX, testY = get_data(testXY)

  sns.set_style('white')

  lda = LDA(nComp=2)
  lda.fit(trainX, trainY)
  X_fs = lda.transform(trainX)

  print("Original Data Size:",trainX.shape, "\nModified Data Size:", X_fs.shape)

  #lda.plot(X_fs)
  x1 = X_fs[:,0]
  x2 = X_fs[:,1]

  plt.scatter(x1, x2, c=trainY, alpha=.8, cmap=plt.cm.get_cmap('coolwarm', 4))
  plt.xlabel('Linear Discriminant 1')
  plt.ylabel('Linear Discriminant 2')
  plt.colorbar()
  plt.title('training data')
  plt.show()
  plt.savefig('proj01_part03_lda_fig01.png', dpi=200)


  lda.fit(testX, testY)
  X_fs = lda.transform(testX)

  print("Original Data Size:",testX.shape, "\nModified Data Size:", X_fs.shape)

  #lda.plot(X_fs)
  x1 = X_fs[:,0]
  x2 = X_fs[:,1]

  plt.scatter(x1, x2, c=testY, alpha=.8, cmap=plt.cm.get_cmap('coolwarm', 4))
  plt.xlabel('Linear Discriminant 1')
  plt.ylabel('Linear Discriminant 2')
  plt.colorbar()
  plt.title('test data')
  plt.show()
  plt.savefig('proj01_part03_lda_fig02.png', dpi=200)




  print('---->>> end of process.')





# EOF
