from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, math
import random as rand
import pickle as pkl
import copy
from pprint import pprint as pp
from pdb import *

from numpy.core.fromnumeric import _trace_dispatcher

# acceptable image formats
exts = {'.png', '.jpg', '.jpeg'}

translations = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", \
  "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", \
  "mucca": "cow", "pecora": "sheep"}

animals=['dog', 'horse','elephant', 'butterfly', 'chicken', 'cat', 'cow', \
 'sheep']

def create_data_obj(src, subdir, categories, n_samples, img_size=250): # size, categories):
  print('---> Creating data object...')
  data = list()
  for category, translate in categories.items():
    path = src+subdir+category+'/'
    target = animals.index(translate)
    print()
    print('   ----------> at %s , %s is a class' % (path, translate))
    for i, file in enumerate(os.listdir(path)):
      if i < n_samples:
        try:
          fname, ext = os.path.splitext(file)
          if prt is True:
            #print()
            print('- %5d  %s%s' % (i,fname,ext))
          image = Image.open(path+fname+ext)
          #image = change_contrast(image, 95) # change contrast for grey scale
          image = ImageOps.grayscale(image)
          image = image.resize((img_size,img_size))
          pixels = np.asarray(image, order='f') # write to memory fortran style
          pixels = pixels.astype('float32')
          data.append(np.asarray([i,pixels,target,src,category,translate,fname+ext]))
        except OSError as error: # if returns error
          print('--> cant read image %s already exists.' % path+fname+ext)
          pass
  return data

def check_dest(dest):
  try:
    os.makedirs(dest, exist_ok=True)
    print('--> output directory %s successfully created.' % dest)
  except OSError as error: # if returns error
    print('--> output directory %s already exists.' % dest)
    shutil.rmtree(dest)
    print('--> output directory %s is removed.' % dest)
    os.makedirs(dest)
    print('--> output directory %s successfully created.' % dest)
  return

def norm_images(data, save, prt=False):
  print()
  for datum in data:
    pixels = datum[1]
    mean, std = pixels.mean(), pixels.std()
    if prt is True:
      print('mean: %.3f, std: %.3f' % (mean, std))
    pixels = pixels/255.0
    mean, std = pixels.mean(), pixels.std()
    if prt is True:
      print('new --> mean: %.3f, std: %.3f' % (mean, std))
      print('    --> min:  %.3f, max: %.3f' % (pixels.min(), pixels.max()))
    datum[1] = pixels
    if save ==True: # for inspection
      image_n = Image.fromarray(pixels)
      dest = datum[3]+'norm-img/'+datum[4]+'/'
      check_dest(dest)
      if image_n.mode != 'L':
        image_n = image_n.convert('L')
      image_n.save(dest+datum[5])
  return data

def change_contrast(image, level=100):
  factor = (259 * (level+255))/(255 *(259-level))
  def contrast (c):
    return 128 + factor * (c-128)
  return image.point(contrast)

def get_data(data, label=None):
  #rand.shuffle(data)
  X = list()
  Y = list()
  for datum in data:
    if label != None:
      if datum[5]==label:
        X.append(datum[1].flatten())
        Y.append(datum[2])
    else:
      X.append(datum[1].flatten())
      Y.append(datum[5])
  X = np.asarray(X, order='f') # write to memory fortran style
  Y = np.asarray(Y, order='f') # write to memory fortran style
  return X,Y

def pca(X):
  print('---> Computing principal components...')
  # get dim
  n_samples, dim = X.shape
  # center data
  mean = X.mean(axis=0)
  X = X - mean
  use_compact_technique = False # disabled, was getting poor results
  if dim > n_samples and use_compact_technique==True:
    # use compact trick
    covar = np.dot(X,X.T) # calculate covariance matrix
    eVals, eVecs = np.linalg.eigh(covar) # get eigen values and vectors of covariance
    compact = np.dot(X.T,eVecs).T # compact trick
    V = compact[::-1] # get last eigenvectors (highest value)
    S = np.sqrt(eVals)[::-1] # get last eigenvalues (highest value)
    for i in range(V.shape[1]):
      V[:,i] /= S  # projection (V) divided by variance (S)
  else:
    # normal method, use SVD
    U,S,V = np.linalg.svd(X)
    V = V[:n_samples] # the rest is not usefull information
  return V, S, mean

def plot_utility(label, image_size, V, img_mean, b, saveImg=True):
  # show images
  fig,axs = plt.subplots(2,5, sharex=True, sharey=True)
  fig.suptitle("Eigen Mean & Features - "+label)
  axs[0,0].imshow(img_mean.reshape(image_size,image_size))
  axs[0,0].set_title('Proj. Mean')
  for i in range(9):
    #plt.subplot(2,5,i+2)
    axs[int((i+1)/5),(i+1)%5].set_title('feat. n°{}'.format(i+1))
    axs[int((i+1)/5),(i+1)%5].imshow(V[i].reshape(image_size,image_size))
  figname = './output/figure_{}_{}.png'.format(b,label)
  if saveImg==True:
    plt.savefig(figname, bbox_inches='tight',dpi=300)
    print('---> Saving class eigen projections figure: '+figname)

  #fig.show(bbox_inches='tight',dpi=100)
  return

class knn_classifier:
  def __init__(self, trainXY, testXY=None, k=1, precision=4, prt=True, vmode='simple', labels=None):
    print('---> Initializing KNN classifier... ')
    self.precision = precision
    self.prt = prt
    self.k = k
    self.voting_mode = vmode
    self.labels = labels # already a list
    self.n_classes = len(self.labels)
    # set up training data
    self.trainXY = trainXY
    self.trainX = copy.deepcopy(trainXY[:][:-1])
    self.trainY = copy.deepcopy(trainXY[:][-1])
    # set up test data
    if testXY is not None:
      self.testXY = testXY
      self.testX = copy.deepcopy(testXY[:][:-1])
      self.testY = copy.deepcopy(testXY[:][-1])
    else:
      print('---> Err: no test set (testX) assigned!!')
    # end of __init__

  def euclidean_dist(self, A, B):
    res = np.sqrt(np.sum((A-B)**2, axis=1))
    return res

  # Calculate nearest neighbors
  def get_kneighbors(self, testX, dist_mode=False): # trainX, trainX_i, # of nearest neighbors
    print('---> Computing distance to K nearest neighbors...')
    knn_dists = list()
    knn_ids = list()
    trainXY = self.trainXY
    k = self.k

    print('------------>> NBUG:{} - dist.shape: {}'.format(__LINE__, row, index))
    set_trace() # check
    # compute distances for given test set
    dist = [self.euclidean_dist(testx_i, trainXY[:][:-1]) for testx_i in testX]
    for row in dist: # sort distances

      index = enumerate(row)
      sorted_knn = sorted(index, key=lambda x: x[1])[:k] # sort tuples based on 1th element
      idList = [tup[0] for tup in sorted_knn]
      distList = [tup[1] for tup in sorted_knn]
      knn_dists.append(distList)
      knn_ids.append(idList)
    set_trace() # check
    if self.prt == True: # print log
      print('')
      print(' TestX ID | KNN IDs (trainXY)        |')
      print('__________|__________________________|')
      for i in range(len(knn_ids)):
        #diststr =' '.join([str(round(item,6)) for item in knn_dists[i] ])
        idsstr =' '.join([str(round(item)) for item in knn_ids[i] ])
        print(' {:5d}  |  {}  |'.format(i , idsstr))
    print()
    if dist_mode:
      return np.array(knn_dists), np.array(knn_ids)
    else:
      return np.array(knn_ids)

  def predict(self, testX, vmode='simple'):
    ''' prediction modes:
          - simple: simple voting among K nearest neighbors.
          - weighted: uses distances as weights in voting process.
    '''
    prt = self.prt
    print('---> Predicting class based on KNN: in {} mode...'.format(vmode))
    if vmode=='weighted':
      probs = list()
      dists, ids = self.get_kneighbors(testX, dist_mode=True)
      inv_dists = 1/dists
      mean_invdist = inv_dists / np.sum(inv_dists, axis=1)[:, np.newaxis]
      set_trace()
      for i, xdatum in enumerate(mean_invdist):
        Y = self.trainY[ids[i]]
        for label in range(self.numLabels):
          ids = np.where(Y==label)
          prob_ids = np.sum(xdatum[ids])
          probs.append(np.array(prob_ids))
      yest_probs = np.array(probs).reshape(testX.shape[0], self.numClasses)
      yest = np.array([np.argmax(p) for p in yest_probs])
      if prt:
        print('--> Predictions: ')
        print(yest)
    else: # in simple mode

      neighbs = self.get_kneighbors(testX)
      yest = np.array([np.argmax(np.bincount(self.trainY[n])) for n in neighbs])
    return yest

  def get_acc(self, testX, testY, vmode='simple'):
    if vmode == 'both': # returns simple yest
      self.prt=True # print outputs
      yest = self.predict(testX, vmode='simple')
      yest_w = self.predict(testX, vmode='weighted')
    else:
      yest = self.predict(testX, vmode=vmode)
    acc = float(sum(yest==testY)) / float(len(testY))
    if self.prt:
      print('--->> Test set prediction accuracy: {}'.format(acc))
    return acc
  # end of knn_classifier class

if __name__ == '__main__':

  ''' ToDo: add a auto downloader
  '''
  #url = 'https://www.kaggle.com/alessiocorrado99/animals10/download'

  ''' Image PCA and KNN
    YouTube: https://www.youtube.com/watch?v=9YOWgQ4kHGg
    Image CPA from scratch:
    https://drscotthawley.github.io/blog/2019/12/21/PCA-From-Scratch.html
    https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
    CPA from scratch: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1
  '''

  ''' Default config. Do not change.
  '''
  skip_process_raw_data = False # Do not change.
  skip_data_load = False # Do not change.
  skip_process_dataXY =  False # Do not change.
  skip_to_eigenXY = False # Do not change.
  prt = False # Do not change.

  ''' NBUG config
  '''
  skip_data_load = True # just to save time
  skip_process_raw_data = True # set to False to save time, set True for first use.
  skip_process_dataXY = True # just to save dev time
  skip_to_eigenXY = True # used saved PCA dataset

  ''' run config
  '''
  prt = True
  image_size = 50 # pixels, equal height and width
  samples_per_class = 1000
  shuffles = 10 # shuffle n times to mix data
  testfrac = .2 # 0-1.0 | test set fraction of main set


  ''' create data object
    data.append(np.asarray([i,pixels,target,src,category,translate,fname+ext]))
  '''
  #data = norm_images(data, save=True) # normalize dataset
  if skip_data_load == False:
    if skip_process_raw_data == False:
      # create data object
      data = create_data_obj(src='./data/', subdir='raw-img/', \
        categories=translations, n_samples=samples_per_class, img_size=image_size)
      # save loaded dataset
      #os.makedirs('temp')
      with open('dataset.json', 'wb') as dataset_file:
        pkl.dump(data, dataset_file)
    else: # load previously loaded dataset
      print('---> Loading dataset object from binary file...')
      with open('dataset.json', 'rb') as dataset_file:
        data = pkl.load(dataset_file)

  ''' get PCA for entire dataset
  '''
  if skip_process_dataXY == False:
    dataXY = list()
    mfeat = image_size**2  # = img_size * img_size + y
    keysXY = np.full((0, mfeat+ 1), 1)
    keyy = np.full((1, mfeat+ 1), 1)
    labels = list()
    eigenXY_class = np.empty((samples_per_class, mfeat+ 1))
    eigenXY = np.empty((0, mfeat+ 1))
    print()
    # get data per class and process to obtain PCA
    for b, (directory, label) in enumerate(translations.items()):
      X, Y = get_data(data, label=label)
      #nSamps, mFeat = len(X), len(X[0])
      V,S,img_mean = pca(X)
      #set_trace()
      dataXY.append([label,X,Y,V,S,img_mean])
      labels.append(label)
      key = np.reshape(img_mean, (1,-1))
      y = np.full((1,1), Y[0])
      # copy processed PCA keys - mean
      keyy = copy.deepcopy(np.concatenate((key,y), axis=1))
      keysXY = copy.deepcopy(np.concatenate((keysXY, keyy), axis=0))
      #set_trace()
      Y = np.expand_dims(Y,axis=1)
      eigenXY_class = copy.deepcopy(np.concatenate((V,Y), axis=1))
      eigenXY = copy.deepcopy(np.concatenate((eigenXY,eigenXY_class), axis=0))
      #print(eigenXY)
      #set_trace()
      # plot or save images
      plot_utility(label, image_size, V, img_mean, b)

    # save processed dataset object
    print('---> Saving dataXY object to binary file...')
    with open('dataXY.json', 'wb') as dataXY_file:
      pkl.dump(dataXY, dataXY_file)

    # save keysXY dataset object
    print('---> Saving keysXY object to binary file...')
    with open('keysXY.json', 'wb') as keysXY_file:
      pkl.dump(keysXY, keysXY_file)

    # save eigenXY dataset object
    print('---> Saving eigenXY object to binary file...')
    with open('eigenXY.json', 'wb') as eigenXY_file:
      pkl.dump(eigenXY, eigenXY_file)

    # save labels dataset object
    print('---> Saving labels object to binary file...')
    with open('labels.json', 'wb') as labels_file:
      pkl.dump(labels, labels_file)

  else:
    if skip_to_eigenXY == False:
      # load processed dataset object
      print('---> Loading dataXY object from binary file...')
      with open('dataXY.json', 'rb') as dataXY_file:
        dataXY = pkl.load(dataXY_file)

      # load keysXY dataset object
      print('---> Loading keysXY object from binary file...')
      with open('keysXY.json', 'rb') as keysXY_file:
        keysXY = pkl.load(keysXY_file)
    else:
      # load eigenXY dataset object
      print('---> Loading eigenXY object from binary file...')
      with open('eigenXY.json', 'rb') as eigenXY_file:
        eigenXY = pkl.load(eigenXY_file)

      # load labels dataset object
      print('---> Loading labels object from binary file...')
      with open('labels.json', 'rb') as labels_file:
        labels = pkl.load(labels_file)


  ''' print labels
  '''
  print()
  print('---> Labels and class indices:')
  for i, label in enumerate(labels):
    print('   | - {:2d} : {}'.format( i, label))
  print()

  ''' keysXY: set of keys for all known classes
  '''
  print('---> Shuffling PCA image dataset...')
  for _ in range(shuffles):
    np.random.shuffle(eigenXY)
  print('   |---> PCA image set shape: {}'.format(eigenXY.shape))

  print('---> Create test set: sampling PCA image dataset w/o replacement...')
  nSamps = eigenXY.shape[0]
  test_size = int(testfrac * nSamps)
  rng = np.random.default_rng()
  testXY = copy.deepcopy(eigenXY[:test_size][:])
  trainXY = copy.deepcopy(eigenXY[test_size:][:])
  print('   |---> PCA image training set shape: {}'.format(trainXY.shape))
  print('   |---> PCA image test set shape: {}'.format(testXY.shape))

  #set_trace()self, trainXY, testXY=None, k=1, precision=4, prt=True, vmode='simple', labels=None):
  knn = knn_classifier(trainXY, testXY, k=5, prt=True, vmode='simple', labels=labels)
  knn.get_acc(knn.testX, knn.testY) #, vmode='simple')
  print('\n---> End of process.')


























































































# EOF
