"""Info
  @author Bardia Mojra
  @date April 13, 2021
  @brief Project 02 on Decision Trees for CSE6363 Machine Learning w Dr. Huber.

  @link https://medium.com/swlh/decision-tree-implementation-from-scratch-in-python-1cff4c00c71f
  @link https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator as opt
import inspect as i
import pdb

"""Globals
"""
eps = np.finfo(float).eps

class decisionTree:
  def __init__(self, XYtrain, maxDepth=None):
    self.depth = 0
    self.maxDepth = maxDepth
    self.X = XYtrain.drop(XYtrain.columns[-1], axis=1)
    self.Y = XYtrain[XYtrain.columns[-1]]
    self.features = np.asarray(self.X.columns)
    self.nFeatures = XYtrain.shape[1]
    self.mSamples = XYtrain.shape[0]
    self.df = self.X.copy()
    self.df['Y'] = self.Y.copy()
    # build decision tree
    self.tree = self.buildTree(self.df)
    self.printTree()
    return

  def buildTree(self, df, tree=None):
    # determine which input feature results in highest infoGain
    feature = self.getBestSplit(df)
    print('best feature: ', feature)
    # init tree
    if tree == None:
      tree = dict()
      tree[feature] = dict()

    if df[feature].dtypes != object: # can add numerical dTree later
      print('\n\n')
      print('df[feature]: ', df[feature])
      print('\n>>>Err: non-object feature at ln:', i.getframeinfo(i.currentframe()).lineno)
      return
    else: # only works with labels
      for val in np.unique(df[feature]): # for each possible value in 'feature' col
        df_ch = self.splitSamples(df, val, feature, opt.eq) # get child subset
        Y_objs, cnts = np.unique(df_ch['Y'], return_counts=True) # return Y class cnts
        #pdb.set_trace()
        if(len(cnts)==1): # single-class, pure group -- Leaf Node
          tree[feature][val] = Y_objs[0]
        else: # impure
          if((self.depth < self.maxDepth) | (self.maxDepth is None)):
            self.depth += 1
            tree[feature][val] = self.buildTree(df_ch)
          if self.maxDepth is not None and self.depth < self.maxDepth:
            tree[feature][val] = Y_objs[np.argmax(cnts)]
          else:

    return tree

  def printTree(self):
    print('\n\n')
    print('-------------------- Decision Tree --------------------')
    print('Tree depth: ', self.maxDepth)
    print(self.tree)
    return

  def splitSamples(self, df, val, col, _opt):
    df_new = df[_opt(df[col], val)] # in df, all in 'col' w 'val' that satisfy '_opt' condition
    df_new = df_new.reset_index(drop=True) # drop old index, reset to num index
    return df_new

  def getTotalEntropy(self, data):
    """Calculates total entropy of the give dataset.
    """
    totalEntropy = 0
    for y in np.unique(data['Y']):
      frac = data['Y'].value_counts()[y] / len(data['Y'])
      totalEntropy += -frac * np.log2(frac)
    return totalEntropy

  def getFeatureEntropy(self, data, a):
    """Calculates entropy per feature for a given dataset, H_D(Y|A).
    """
    entropy = 0
    #threshold = None # for numeric features
    if data[a].dtypes == object: # make sure datatype is what we expect
      for val in np.unique(data[a]):  # sum of H_D(Y|A=a)
        featureEntropy = 0
        for y in np.unique(data['Y']):  # add all per datum feature entropies
          num = len(data[a][data[a] == val][data['Y'] == y])
          den = len(data[a][data[a] == val])
          infoGain = num / (den + eps)  # information gain
          if infoGain > 0:
            featureEntropy += -infoGain * np.log2(infoGain)
        featureWeight = len(data[a][data[a] == val]) / len(data)
        entropy += featureWeight * featureEntropy
    else: # else could be numeric data
      print('>>>Err: none object data at ln:', i.getframeinfo(i.currentframe()).lineno)
    return entropy

  def getBestSplit(self, df):
    """
      For a given dataset, it return the feature with highest information gain.
      InfoGain = Entropy(data) - Sum of Entropy(data_subsets)
      -> IG_D(Y|A) = H_D(Y) - H_D(Y|A) ; where D is given data and A is some input
      feature.
      Entropy =
      Sum of all Entropy(data_subsets) =
    """
    infoGain = list()
    igSum = 0.0
    parentEntropy = self.getTotalEntropy(df) # H_D(Y)
    print('\nparentEntropy:{:.5f}'.format(parentEntropy))
    for a in list(df.columns[:-1]):
      featureEntropy = self.getFeatureEntropy(df, a) # H_D(Y\A=a)
      infoGain_a = parentEntropy - featureEntropy
      igSum += infoGain_a
      infoGain.append(infoGain_a)
      print('feature:{1:>5s}'.format(4, a), '  infoGain:{:.5f}'.format(infoGain_a))
    print('Sum of infoGains: {:.5f}'.format(igSum))
    return df.columns[:-1][np.argmax(infoGain)]

  def y_est(self, xDatum, features, tree):
    for node in tree.keys():
      val = xDatum[node]
      tree = tree[node][val]
      if type(tree) is dict:
        pred = self.y_est(xDatum, features, tree)
      else:
        pred = tree
        return pred
    return pred

  def getEst(self, X):
    predictions  = list()
    features = {label: i for i, label in enumerate(list(X.columns))}
    for idx in range(len(X)):
      predictions.append(self.y_est(X.iloc[idx], features, self.tree))
    return predictions
  # end of decisionTree class

def getAcc(gndTruth, Est):
  correct = 0
  for i in range(len(gndTruth)):
    if gndTruth[i] == Est[i]:
      correct += 1
  return correct / float(len(gndTruth)) * 100.0

def getData(XY):
  X = XY.drop(XY.columns[-1], axis=1)
  Y = XY[XY.columns[-1]]
  return X, Y

"""Main
"""
if __name__ == "__main__":

  # import data
  XYtrain = pd.read_csv("./tic-tac-toe_train.csv")
  XYtest = pd.read_csv("./tic-tac-toe_test.csv")

  # test
  print(XYtrain.head())

  dTree = decisionTree(XYtrain, maxDepth=2)
  Xtest, Ytest = getData(XYtest)
  Y_est = dTree.getEst(Xtest)
  acc = getAcc(Ytest, Y_est)
  print('Training accuracy: ', acc)
  print()
