W = 'W'
M = 'M'


trainData = [(170, 57, 32, W), (190, 95, 28, M), (150, 45, 35, W),
(168, 65, 29, M), (175, 78, 26, M), (185, 90, 32, M), (171, 65, 28, W),
(155, 48, 31, W), (165, 60, 27, W), (182, 80, 30, M), (175, 69, 28, W),
(178, 80, 27, M), (160, 50, 31, W), (170, 72, 30, M)]

testX = {(162, 53, 28, W), (168, 75, 32, M), (175, 70, 30, W), (180, 85, 29, M)}

testX_noAge = {(162, 53), (168, 75), (175, 70), (180, 85)}





from random import randrange
from math import exp
from random import seed
import pdb

PRECISION = 1000

import pprint
from pprint import pprint
# Find the min and max values for each column
def dataset_minmax(dataset):
  minmax = list()
  for i in range(len(dataset[0])-1):
    col_values = [row[i] for row in dataset]
    value_min = min(col_values)
    value_max = max(col_values)
    minmax.append([value_min, value_max])
  pprint(minmax)
  return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
  normList = list()
  for row in dataset:
    for i in range(len(row)-1):
      newrow = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    normList.append(newrow)
  return normList


def get_acc(groundtruth, prediction):
  correct = 0
  for i in range(len(groundtruth)):
    if groundtruth[i] == prediction[i]:
      correct += 1
  return correct / float(len(groundtruth)) * 100.0

def sgd(XY, alpha, n):
  pdb.set_trace()
  theta = [0.0 for i in range(len(XY))]
  for _ in range(n):
    for datum in XY:
      Y_ = hypothesis(datum, theta)
      e = datum[-1] - Y_
      theta[0] = theta[0] + alpha * e * Y_ * (1.0 - Y_)
      for i in range(len(datum)-1):
        theta[i + 1] = theta[i + 1] + alpha * e * Y_ * (1.0 - Y_) * datum[i]
  return theta

def hypothesis(X, theta):
  y_ = theta[0]
  for i in range(len(X)-1):
    y_ += theta[i+1] * X[i]
  return softmax(y_)

def softmax(y_):
  return 1.0/(1.0 + exp(-y_))

def rsig(y_):
  return round(y_ * PRECISION)/PRECISION

def prepare_data(data):

  #pdb.set_trace()
  return data

if __name__ == '__main__':
  pdb.set_trace()
  n = 100
  alpha = .1
  data = prepare_data(trainData)
  XY = normalize_dataset(data, dataset_minmax(data))
  Xt = testX
  predictions = list()
  theta = sgd(XY, alpha, n)

  for x in Xt:
    y_ = hypothesis(x, theta)
    y_ = rsig(y_)
    predictions.append(y_)

  y = [xy[-1] for xy in XY]
  acc = get_acc(y,y_)
  print("acc: ", acc)
