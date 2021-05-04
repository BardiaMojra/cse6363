import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.nanfunctions import _nanmedian_small
from sklearn.datasets import make_blobs


D= {(170,57,32),(190,95,28),(150,45,35),(168,65,29),(175,78,26),(185,90,32),
(171,65,28),(155,48,31),(165,60,27),(182,80,30),(175,69,28),(178,80,27),
(160,50,31),(170,72,30)}




import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


def get_data(data, Ymax=None,  Ymin=None):
  #Y = [Ymax if i > 0 else Ymin for i in Y]
  Y = np.asarray(Y)
  Y = np.expand_dims(Y,axis=1)
  XY = np.concatenate((X, Y), axis=1)
  print('dataset: ')
  print(XY)
  print('\n\n')
  return X, Y





if __name__ == '__main__':

  # create a blob of 200 data points
  dataset = make_blobs(n_samples=200,
                      n_features=2,
                      centers=4,
                      cluster_std = 1.6,
                      random_state=50)
  points = dataset[0]
  D = get_data(D)
  #d1 = D[0:1]


  # create dendrogram
  dendrogram = sch.dendrogram(sch.linkage(D, method='ward'))

  print(D)

  plt.show(dendrogram)

  D_sp = plt.scatter()


  #print('-->> Training and testing with OR')
  #X, Y = get_data(dataOR, Ymax=1, Ymin=0) # rescale output to [0,1] since we're using Sigmoid activation function

  #nnOR = nn_2layer(X, Y, lr=.001, iters=1000)
  #nnOR.BGD()

  #pdb.set_trace()

  #plt.plot(range(len(nnOR.log.loss)), nnOR.log.loss, label='Network Batch Loss')
  #plt.show()
  #figOR = plt.figure()
