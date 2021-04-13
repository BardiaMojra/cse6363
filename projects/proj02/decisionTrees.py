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
import math
import operator
import inspect as i

import pdb

"""Globals
"""
eps = np.finfo(float).eps

class decisionTree:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    return

  def getTotalEntropy(self, data):
    """Calculates total entropy of the give dataset.
    """
    totalEntropy = 0
    for y in np.unique(data["y"]):
      frac = data["y"].value_counts()[y] / len(data["y"])
      totalEntropy += -frac * np.log2(frac)
    self.totalEntropy = totalEntropy
    return totalEntropy

  def getFeatureEntropy(self, data, a):
    """Calculates entropy per feature for a given dataset, H_D(Y|A).
    """
    totalEntropy = 0
    threshold = None
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
    else:
      print('>>>Err: none object data at ln:', i.getframeinfo(i.currentframe()).lineno)


  def getBestSplit(self, data):
    """
    For a given dataset, it return the feature with highest information gain.
    InfoGain = Entropy(data) - Sum of Entropy(data_subsets)
    -> IG_D(Y|f) = H_D(Y) - H_D(Y|A) ; where D is given data and A is some input
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



	return df.columns[:-1][np.argmax(ig)], thresholds[np.argmax(ig)] #Returns feature with max information gain



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
