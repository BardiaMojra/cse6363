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
  def __init__(self, XYtrain, maxDepth):
    self.maxDepth = maxDepth
    self.X = XYtrain.drop(XYtrain.columns[-1], axis=1)
    self.Y = XYtrain[XYtrain.columns[-1]]
    self.features = np.asarray(self.X.columns)
    self.nFeatures = XYtrain.shape[1]
    self.mSamples = XYtrain.shape[0]
    self.df = self.X.copy()
    self.df['Y'] = self.Y.copy()


    pdb.set_trace()
    # build decision tree
    self.tree = self.buildTree(self.df)
    self.printTree()

    return

  def buildTree(self, df, tree=None):

    # determine which feature results in highest infoGain
    feature = self.getBestSplit(df)

    # init tree
    if tree == None:
      tree = list()
      tree[feature] = list()

    if df[feature].dtypes is not object:
      print('\n>>>Err: non-object feature at ln:', i.getframeinfo(i.currentframe()).lineno)
      return


    for datum in df[feature]:
      childDF = self.splitSamples(df, datum, feature, opt.eq)


  def printTree(self):

  def train(self, X, Y):
    self.inputFeatures = list()

  def getTotalEntropy(self, data):
    """Calculates total entropy of the give dataset.
    """
    totalEntropy = 0
    for y in np.unique(data["y"]):
      frac = data["y"].value_counts()[y] / len(data["y"])
      totalEntropy += -frac * np.log2(frac)
    return totalEntropy

  def getFeatureEntropy(self, data, a):
    """Calculates entropy per feature for a given dataset, H_D(Y|A).
    """
    totalEntropy = 0
    threshold = None # for numeric features
    if data[a].dtypes == object: # make sure datatype is what we expect
      for featureValue in np.unique(data[a]):  # sum of H_D(Y|A=a)
        featureEntropy = 0
        for y in np.unique(data["y"]):  # add all per datum feature entropies
          num = len(data[a][data[a] == featureValue][data["y"] == y])
          den = len(data[a][data[a] == featureValue])
          infoGain = num / (den + eps)  # information gain
          if infoGain > 0:
            featureEntropy += -infoGain * np.log2(infoGain)
        featureWeight = len(data[a][data[a] == featureValue]) / len(data)
        totalEntropy += featureWeight * featureEntropy
    else: # else could be numeric data
      print('>>>Err: none object data at ln:', i.getframeinfo(i.currentframe()).lineno)

  def getBestSplit(self, data):
    """
      For a given dataset, it return the feature with highest information gain.
      InfoGain = Entropy(data) - Sum of Entropy(data_subsets)
      -> IG_D(Y|A) = H_D(Y) - H_D(Y|A) ; where D is given data and A is some input
      feature.
      Entropy =
      Sum of all Entropy(data_subsets) =
    """
    infoGain = list()
    thresholds = list()
    parentEntropy = self.getTotalEntropy(data) # H_D(Y)

    for a in list(data.columns[-1]):
      featureEntropy, threshold = self.getFeatureEntropy(data, a) # H_D(Y\A=a)
      infoGain_a = parentEntropy - featureEntropy
      infoGain.append(infoGain_a)
      thresholds.append(threshold)
    return data.columns[:-1][np.argmax(infoGain)], thresholds[np.argmax(infoGain)]





"""Main
"""
if __name__ == "__main__":

  # import data
  XYtrain = pd.read_csv("./tic-tac-toe_train.csv")
  XYtest = pd.read_csv("./tic-tac-toe_test.csv")

  # test
  print(XYtrain.head())

  X = XYtrain.drop(XYtrain.columns[-1], axis=1)
  Y = XYtrain[XYtrain.columns[-1]]

  dTree = decisionTree(XYtrain, maxDepth=2)
  #dTree.train()
