
from mmap import MADV_NORMAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix


def get_acc(yest, testY, test_label='_'):
  #set_trace()
  acc = float(sum(yest==testY)) / float(len(testY))
  print('-->> {} acc: {:.3f}%'.format(test_label, (acc*100)))
  return acc


if __name__ == '__main__':
  ''' Config
  '''
  label_pLimit = 0.60


  # prep training data
  print('Ds:')
  trainXY = pd.read_csv('Ds.csv')
  trainX = trainXY.drop('gender', axis=1)
  trainY = trainXY.gender
  print('trainXY: {}'.format(trainXY.shape))
  print('trainX: {}'.format(trainX.shape))
  print('trainY: {}'.format(trainY.shape))
  print()

  # unlabeled data
  uX = pd.read_csv('Du.csv')
  print('uX: {}'.format(uX.shape))
  print()

  # prep test data
  print('Dt:')
  testXY = pd.read_csv('Dt.csv')
  testX =testXY.drop('gender', axis=1)
  testY =testXY.gender
  print('testXY: {}'.format(testXY.shape))
  print('testX: {}'.format(testX.shape))
  print('testY: {}'.format(testY.shape))
  print()

  # visualize class distro
  trainY.value_counts().plot(kind='bar')
  #plt.xticks(['M', 'W'], ['Male', 'Female'])
  plt.ylabel('Count')
  #plt.show()
  plt.savefig('p02_fig01_Class_distribution.png', dpi=200)

  logreg = LogisticRegression(max_iter=100)
  logreg.fit(trainX, trainY)
  yest_test = logreg.predict(testX)
  yest_train = logreg.predict(trainX)

  print('Simple Logistic Regression: ')
  get_acc(yest_train, trainY, 'training data')
  get_acc(yest_test, testY, 'test data')
  print()

  plot_confusion_matrix(logreg, testX,testY, cmap='Blues', normalize='true' )
  #plt.show()
  plt.savefig('p02_fig02_Class_distribution.png', dpi=200)

  print('Probabilities of TestX Predictions:')
  logreg.predict_proba(testX)


  ''' Self-training:
    1. Train LogReg on labeled data (TrainXY).
    2. Use classifier to predict unlabeled data, as well as the probabilities.
    3. Concatenate the 'pseudo-labeled' dateset with labeled training data and
    retrain the classifier with it.
    4. Evaluate on test date.
  '''

  iters = 0

  # new accuracy hist for pseudo-labeled data
  train_acc_hist = list()
  test_acc_hist = list()
  uX_yest_hist = list()


  # init with max probability
  high_probs = [1]
  while len(high_probs)>0: # loop over high probability values till there is no more
    # train on labeled data and acc on training and test data
    logreg = LogisticRegression(max_iter=100)
    logreg.fit(trainX, trainY)
    yest_train = logreg.predict(trainX)
    yest_test = logreg.predict(testX)

    print()
    print('Logistic Regression: -- iter: {}'.format(iters))
    train_acc = get_acc(yest_train, trainY, 'training data')
    test_acc = get_acc(yest_test, testY, 'test data')
    train_acc_hist.append(train_acc)
    test_acc_hist.append(test_acc)

    print()
    print('Predict unlabeled...')
    Uest_probs = logreg.predict_proba(uX)
    Uest = logreg.predict(uX)
    prob0 = Uest_probs[:,0]
    prob1 = Uest_probs[:,1]

    # save probs in a df
    Uest_probs_hist = pd.DataFrame([])
    Uest_probs_hist['Uest'] = Uest
    Uest_probs_hist['Uest_Porb0'] = prob0
    Uest_probs_hist['Uest_Porb1'] = prob1
    Uest_probs_hist.index = uX.index

    print()
    print('unlabeled data predictions and probabilities: ')
    pp(Uest_probs_hist)

    # keep high probability labels
    high_probs = pd.concat([Uest_probs_hist.loc[Uest_probs_hist['Uest_Porb0'] > label_pLimit],
                            Uest_probs_hist.loc[Uest_probs_hist['Uest_Porb1'] > label_pLimit]],
                          axis=0)
    print()
    print('high probability labels: ')
    pp(high_probs)

    # append new datum - labeled
    trainX = pd.concat([trainX, uX.loc[high_probs.index]], axis=0)
    trainY = pd.concat([trainY, high_probs.Uest])

    # drop newly labeled
    uX = uX.drop(index=high_probs.index)
    print(f"{len(uX)} unlabeled instances remaining.\n")

    print()
    print('remaining predictions with pseudo labels:')
    pp(Uest_probs_hist)

    print()
    print('high probability threshold: {}'.format(label_pLimit))
    print()


    iters += 1


  print('End of process. ----->>> ')


  # EOF
