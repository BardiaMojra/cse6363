'''
  @author Bardia Mojra

  @link https://github.com/ZwEin27/Hierarchical-Clustering/blob/master/hclust.py
  @link https://towardsdatascience.com/a-gentle-introduction-to-self-training-and-semi-supervised-learning-ceee73178b38
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import os
import heapq
import itertools
import pdb




def get_data(data, label):
  Xlist = list()
  for x in data:
    Xlist.append(np.asarray(x))
  X = np.asarray(Xlist)
  print(label+':')
  print(X)
  print('\n')
  return X

'''
def euclidean_distance(self, row_A, row_B):
  dist = 0.0
  diffList = list()
  for i in range(len(row_A[0])):
    diff = 0.0
    diff = row_A[0][i]-row_B[i]
    diffList.append(diff)
    dist += (diff)**2
  dist = math.sqrt(dist)
  return round(dist, self.precision)
'''

class Hierarchical_Clustering:
  def __init__(self, ipt_data, k, linkage='single'):
    self.input_file_name = ipt_data
    self.k = k
    self.dataset = None
    self.linkage = linkage
    self.dataset_size = 0
    self.dimension = 0
    self.heap = []
    self.clusters = []
    self.gold_standard = {}

  def initialize(self):
    """ Initialize and check parameters
    """
    # check file exist and if it's a file or dir
    if not os.path.isfile(self.input_file_name):
      self.quit("Input file doesn't exist or it's not a file")
    self.dataset, self.clusters = self.load_data(self.input_file_name)
    self.dataset_size = len(self.dataset)
    if self.dataset_size == 0:
      self.quit("Input file doesn't include any data")
    if self.k == 0:
      self.quit("k = 0, no cluster will be generated")
    if self.k > self.dataset_size:
      self.quit("k is larger than the number of existing clusters")
    self.dimension = len(self.dataset[0]["data"])
    if self.dimension == 0:
      self.quit("dimension for dataset cannot be zero")


  def euclidean_distance(self, data_point_one, data_point_two):
    """
    euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    assume that two data points have same dimension

    """
    size = len(data_point_one)
    result = 0.0
    for i in range(size):
      f1 = float(data_point_one[i])   # feature for data one
      f2 = float(data_point_two[i])   # feature for data two
      tmp = f1 - f2
      result += pow(tmp, 2)
    result = math.sqrt(result)
    return result

  def compute_pairwise_distance(self, dataset):
    result = []
    dataset_size = len(dataset)
    for i in range(dataset_size-1):    # ignore last i
      for j in range(i+1, dataset_size):     # ignore duplication
        dist = self.euclidean_distance(dataset[i]["data"], dataset[j]["data"])

        # duplicate dist, need to be remove, and there is no difference to use tuple only
        # leave second dist here is to take up a position for tie selection
        result.append( (dist, [dist, [[i], [j]]]) )
    return result

  def build_priority_queue(self, distance_list):
    if self.linkage == 'single':
      heapq.heapify(distance_list)
    elif self.linkage == 'complete':
      heapq._heapify_max(distance_list)
    else:
      print("error - heapq not properly assigned.")
    self.heap = distance_list
    return self.heap

  def compute_centroid_two_clusters(self, current_clusters, data_points_index):
    size = len(data_points_index)
    dim = self.dimension
    centroid = [0.0]*dim
    for index in data_points_index:
      dim_data = current_clusters[str(index)]["centroid"]
      for i in range(dim):
        centroid[i] += float(dim_data[i])
    for i in range(dim):
      centroid[i] /= size
    return centroid

  def compute_centroid(self, dataset, data_points_index):
    size = len(data_points_index)
    dim = self.dimension
    centroid = [0.0]*dim
    for idx in data_points_index:
      dim_data = dataset[idx]["data"]
      for i in range(dim):
        centroid[i] += float(dim_data[i])
    for i in range(dim):
      centroid[i] /= size
    return centroid

  def hierarchical_clustering(self):
    """
    Main Process for hierarchical clustering
    """
    dataset = self.dataset
    current_clusters = self.clusters
    old_clusters = []
    heap = hc.compute_pairwise_distance(dataset)
    heap = hc.build_priority_queue(heap)

    while len(current_clusters) > self.k:
      if self.linkage == 'single':
        dist, m_item = heapq.heappop(heap)  # get min distance
      elif self.linkage == 'complete':
        dist, m_item = heapq._heappop_max(heap)  # get max distance
      else:
        print("error - heapq not properly assigned.")

      # pair_dist = m_item[0]
      pair_data = m_item[1]
      #pdb.set_trace()
      # judge if include old cluster
      if not self.valid_heap_node(m_item, old_clusters):
        continue

      print('> merging '+str(m_item[1][0][0])+' & '+str(m_item[1][1][0]) \
        +' with distance of '+str(rnd(m_item[0], 3)))
      new_cluster = {}
      new_cluster_elements = sum(pair_data, [])
      new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
      new_cluster_elements.sort()
      new_cluster.setdefault("centroid", new_cluster_cendroid)
      new_cluster.setdefault("elements", new_cluster_elements)
      for pair_item in pair_data:
        old_clusters.append(pair_item)
        del current_clusters[str(pair_item)]
      self.add_heap_entry(heap, new_cluster, current_clusters)
      current_clusters[str(new_cluster_elements)] = new_cluster
    current_clusters = sorted(current_clusters)
    return current_clusters

  def valid_heap_node(self, heap_node, old_clusters):
    pair_dist = heap_node[0]
    pair_data = heap_node[1]
    for old_cluster in old_clusters:
      if old_cluster in pair_data:
        return False
    return True

  def add_heap_entry(self, heap, new_cluster, current_clusters):
    for ex_cluster in current_clusters.values():
      new_heap_entry = []
      dist = self.euclidean_distance(ex_cluster["centroid"], new_cluster["centroid"])
      new_heap_entry.append(dist)
      new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
      heapq.heappush(heap, (dist, new_heap_entry))

  def evaluate(self, current_clusters):
    gold_standard = self.gold_standard
    current_clustes_pairs = []
    precision  = int
    recall = int
    for a in range(len(current_clusters)):
      va = current_clusters.__getitem__(a)
      tmp = list(itertools.combinations(get_cluster(va), 2))
      current_clustes_pairs.extend(tmp)

    tp_fp = len(current_clustes_pairs)

    gold_standard_pairs = []

    for gold_standard_value in gold_standard.items():
      tmp = list(itertools.combinations(gold_standard_value, 2))
      gold_standard_pairs.extend(tmp)
    tp_fn = len(gold_standard_pairs)
    tp = 0.0

    pdb.set_trace()


    for ccp in current_clustes_pairs:
      if ccp in gold_standard_pairs:
        tp += 1

    if tp_fp == 0:
      precision = 0.0
    else:
      precision = tp/tp_fp

    if tp_fn == 0:
      precision = 0.0
    else:
      recall = tp/tp_fn
    return precision, recall

  ''' Helper Functions
  '''
  def load_data(self, input_file_name):
    """
    load data and do some preparations

    """
    input_file = open(input_file_name, 'r')
    dataset = []
    clusters = {}
    gold_standard = {}
    id = 0
    for line in input_file:
      line = line.strip('\n')
      row = str(line)
      row = row.split(",")
      target_class = row[-1]

      data = {}
      data.setdefault("id", id)   # duplicate
      data.setdefault("data", row[:-1])
      data.setdefault("class", row[-1])
      dataset.append(data)

      clusters_key = str([id])
      clusters.setdefault(clusters_key, {})
      clusters[clusters_key].setdefault("centroid", row[:-1])
      clusters[clusters_key].setdefault("elements", [id])

      gold_standard.setdefault(target_class, [])
      gold_standard[target_class].append(id)
      id += 1
    return dataset, clusters #, gold_standard

  def quit(self, err_desc):
    raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')

  def loaded_dataset(self):
    """
    use for test only
    """
    return self.dataset

  def display(self, current_clusters): #, precision, recall):
    print()
    #print(' --> Cluster Evaluation:')
    #print(precision)
    #print(recall)
    print(' --> Final clusters:')
    clusters = current_clusters
    for cluster in clusters:
      print(cluster)

'''
def get_clusters(clus_set):
  return [clusters.append(get_clusters(cluster)) for cluster in clus_set]

def get_cluster(cluster):
  clus = cluster.replace('[','')
  clus = clus.replace(']','')
  clus = clus.replace(' ','')
  clusList = clus.split(',')
  clusList = [ int(n) for n in clusList]
  return clusList
'''
def get_clusters(clus_set):
  return [get_cluster(cluster) for cluster in clus_set]

def get_cluster(cluster):
  clus = cluster.replace('[','')
  clus = clus.replace(']','')
  clus = clus.replace(' ','')
  clusList = clus.split(',')
  clusList = [ int(n) for n in clusList]
  return clusList

def rnd(num, precision):
  return math.floor(num * 10**precision)/10**precision




'''main
'''
if __name__ == '__main__':

  Ds = get_data(Ds, 'training dataset')
  Ys = get_data(Ds, 'training target')
  Ys = Ys[:, -1]
  fset_1 = Ds[:, [0,1]]
  fset_2 = Ds[:, [0,2]]
  fset_3 = Ds[:, [1,2]]

  figure, (plt1, plt2, plt3) = plt.subplots(1,3)
  plt1.scatter(fset_1[:,0], fset_1[:,1], color='b')
  plt1.set_xlabel('Height')
  plt1.set_ylabel('Weight')

  plt2.scatter(fset_2[:,0], fset_2[:,1], color='r')#, title='Height vs. Age')
  plt2.set_xlabel('Height')
  plt2.set_ylabel('Age')

  plt3.scatter(fset_3[:,0], fset_3[:,1], color='g')#, title='Weight vs. Age')
  plt3.set_xlabel('Weight')
  plt3.set_ylabel('Age')

  figure.suptitle('Feature Correlations')
  plt.savefig('./p01_fig01_feature_correlations.png')
  figure.show()


  XYs = './Ds.dat'
  colors = ['b', 'r', 'g', 'c', 'm']
  print()
  print()
  print('>>>>> minimum or single linkage hierarchial clustering <<<<<')

  figure_hcmin, (plthcmin1, plthcmin2) = plt.subplots(1,2)
  figure_hcmin.suptitle('Min or Single Linkage HC')
  #pdb.set_trace()

  # plot the groundtruth data
  for x in range(len(fset_1)):
    if Ys[x] == 'M': color = 'r'
    else: color = 'b'
    plthcmin1.scatter(fset_1[x,0], fset_1[x,1], color=color)
  plthcmin1.set_title('Ground-Truth')
  #plthcmin1.legend(loc='upper right', shadow=True)
  plthcmin1.set_ylabel('Weight')
  plthcmin1.set_xlabel('Height')

  #pdb.set_trace()

  k=2
  print()
  print(' |--> k:', k)
  # make the clusters
  hc = Hierarchical_Clustering(XYs, k)
  hc.initialize()
  current_clusters = hc.hierarchical_clustering()
  #precision, recall = hc.evaluate(current_clusters)
  hc.display(current_clusters)#, precision, recall)
  clusters = get_clusters(current_clusters)
  # plot the clusters generated in previous step
  for i, cluster in enumerate(clusters):
    color = colors[i]
    for n in cluster:
      plthcmin2.scatter(fset_1[n,0], fset_1[n,1], color=color)
  plthcmin2.set_title('k:2')
  plthcmin2.set_xlabel('Height')
  figure_hcmin.show()
