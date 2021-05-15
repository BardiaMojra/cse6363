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

#animals=['dog', 'horse','elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep']

animals=['dog', 'horse', 'sheep', 'chicken']

def create_data_obj(src, subdir, categories, n_samples, img_size=250): # size, categories):
  print('---> Creating data object...')
  data = list()
  print()
  for category, translate in categories.items():
    if translate in animals:
      path = src+subdir+category+'/'
      target = animals.index(translate)
      print('---> at %s, %s is a class' % (path, translate))
      for i, file in enumerate(os.listdir(path)):
        if i < n_samples:
          try:
            fname, ext = os.path.splitext(file)
            #if prt is True:
              #print()
              #print(' - %5d  %s%s' % (i,fname,ext))
            image = Image.open(path+fname+ext)
            #image = change_contrast(image, 95) # change contrast for grey scale
            image = ImageOps.grayscale(image)
            image = image.resize((img_size,img_size))
            #set_trace()
            pixels = np.asarray(image, order='f') # write to memory fortran style
            pixels = pixels.astype('float32')
            data.append(np.asarray([i,pixels,target,src,category,translate,fname+ext], dtype=object))
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

def pca(X, components):
  print('---> Computing principal components...')
  assert X.shape[0]!=0, '   |---> Err: selected dataset is emptry...'
  #print('   |---> X.shape: ', X.shape)

  mean = X.mean(axis=0) # normalize data
  std = np.std(X, axis=0) + 0.001 # increase std for potential noise
  Xm = (X - mean)/std
  U,S,V = np.linalg.svd(Xm, full_matrices=True)
  #set_trace()
  V = V[:,:components] # the rest is not usefull information
  #print('   |---> V.shape: ', V.shape)
  #print('   |---> Xm.shape: ', Xm.shape)
  Z = Xm.dot(V)
  mean = Z.mean(axis=0) # normalize PCA data
  return V, S, mean, Z

def plot_utility(label, image_size, Z, img_mean, b, numPC, saveImg=True):
  # show images
  fig,axs = plt.subplots(2,5, sharex=True, sharey=True)
  fig.suptitle('PC Features ({}) - {}, Img:{}x{}'.format(numPC,label,image_size,image_size))
  #img = np.expand_dims(img_mean,axis=1)
  #axs[0,0].imshow(img.reshape(image_size,image_size))
  #axs[0,0].set_title('PC mean')
  for i in range(10):
    #plt.subplot(2,5,i+2)
    axs[int((i)/5),(i)%5].set_title('PC n°{}'.format(i+1))
    axs[int((i)/5),(i)%5].imshow(Z[i].reshape(image_size,image_size))
  tag = get_tag(clen, b, label, image_size, numPC)
  figname = './output/figure_{}'.format(tag)
  if saveImg==True:
    plt.savefig(figname, bbox_inches='tight',dpi=300)
    print('---> Saving class PC projections figure: '+figname)
  #fig.show(bbox_inches='tight',dpi=100)
  return

def plot_PCvariance(label, image_size, b, numPC, saveImg=True):
  with plt.style.context('seaborn'):
    plt.figure()
    plt.title('Principle Component Variances: {}, Img:{}x{}'.format(label,image_size,image_size))
    plt.xlabel('Principle Components')
    plt.ylabel('Variance')
    plt.plot(S)
    #plt.legend('')
    tag = get_tag(clen, b, label, image_size, numPC)
    figname = './output/figure_PCvar_{}'.format(tag)
    if saveImg==True:
      plt.savefig(figname, bbox_inches='tight',dpi=300)
      print('---> Saving class PC variance figure: '+figname)
    #fig.show(bbox_inches='tight',dpi=100)
  return


def get_tag(clen, b, label, image_size, PCr, s='_'):
  clen = 'CL'+str(clen)
  b = s+'C'+str(b)
  label = s+str(label)
  image_size = s+'Res'+str(image_size)
  PCr = s+'PCr'+str(PCr)
  tag = clen+b+label+image_size+PCr
  #print('   --->>> Test ID/config tag: '+tag)
  return tag

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
    self.trainX = copy.deepcopy(trainXY[:,:-1])
    self.trainY = copy.deepcopy(trainXY[:,-1].astype(int))
    # set up test data
    if testXY is not None:
      self.testXY = testXY
      self.testX = copy.deepcopy(testXY[:,:-1])
      self.testY = copy.deepcopy(testXY[:,-1])
    else:
      print('---> Err: no test set (testX) assigned!!')
    # end of __init__

  def euclidean_dist(self, A, B):
    res = np.sqrt(np.sum((A-B)**2, axis=1))
    #set_trace()
    return res

  # Calculate nearest neighbors
  def get_kneighbors(self, testX, dist_mode=False): # trainX, trainX_i, # of nearest neighbors
    if prt:
      print('---> Computing distance to K nearest neighbors...')
    print_all = False
    dist = list()
    knn_dists = list()
    knn_ids = list()
    knn_cls_ids = list()
    k = self.k
    #set_trace() # check
    # compute distances for given test set
    for testx_i in testX:
      testx = np.expand_dims(testx_i, axis=0)
      #print('testx: {}'.format(testx.shape))
      d = self.euclidean_dist(testx, self.trainX)
      #set_trace() # check
      dist.append(d)
    for row in dist: # sort distances
      #set_trace() # check
      index = enumerate(row)
      sorted_knn = sorted(index, key=lambda x: x[1])[:k] # sort tuples based on 1th element
      idList = [tup[0] for tup in sorted_knn]
      distList = [tup[1] for tup in sorted_knn]
      gtList = [ self.trainY[tup[0]] for tup in sorted_knn]
      knn_dists.append(distList)
      knn_ids.append(idList)
      knn_cls_ids.append(gtList)
      if print_all:# NBUG
        print('idList: {}'.format(str(idList)))
        print('distList: {}'.format(str([round(i,self.precision) for i in distList])))
        print('gtList: {}'.format(str(gtList)))
    knn_ids = np.asarray(knn_ids)
    knn_dists = np.asarray(knn_dists)
    knn_cls_ids = np.asarray(knn_cls_ids)
    #set_trace() # check
    if show_knn_details == True:
      self.print_neighbors(knn_ids, knn_cls_ids)
    if dist_mode:
      return knn_dists, knn_ids
    else:
      return knn_ids

  def predict(self, testX, vmode='simple'):
    ''' prediction modes:
          - simple: simple voting among K nearest neighbors.
          - weighted: uses distances as weights in voting process.
    '''
    prt = False
    if prt:
      print('---> Predicting class based on KNN: in {} mode...'.format(vmode))
    if vmode=='weighted':
      probs = list()
      dists, ids = self.get_kneighbors(testX, dist_mode=True)
      inv_dists = 1/dists
      m_inv_dists = inv_dists / np.sum(inv_dists, axis=1)[:, np.newaxis]
      #set_trace()
      for i, row in enumerate(m_inv_dists):
        row_est = self.trainY[ids[i]]
        #print('row_est: {}'.format(str(row_est)))
        #set_trace()
        for v in np.arange(0, len(self.labels)):
          indx = np.where(v==row_est)
          prob_idx = np.sum(row[indx])
          probs.append(np.array(prob_idx))
          #set_trace()
          #if prt:# NBUG
            #print('indx: {}'.format(str(indx)))
            #print('prob_idx: {}'.format(str(prob_idx)))
            #print('probs: {}'.format(str(probs)))
      yest_probs = np.array(probs).reshape(testX.shape[0], len(self.labels))
      yest = np.array([np.argmax(p) for p in yest_probs])
      #set_trace()
    else: # in simple mode
      neighbors = self.get_kneighbors(testX)
      listtemp = list()
      for n in neighbors:
        #set_trace()
        tmp = np.argmax(np.bincount(self.trainY[n]))
        listtemp.append(tmp)
      yest = np.array(listtemp)
    if prt:
      print()
      print('---> Set predictions: ')
      print(yest)
      print('---------->>>')
    return yest

  def get_acc(self, testX, testY, vmode='simple', test='_'):
    yest = self.predict(testX, vmode=vmode)
    #set_trace()
    acc = float(sum(yest==testY)) / float(len(testY))
    self.print_est(testX, testY, yest, vmode, acc, test=test)
    return acc

  def print_neighbors(self, knn_ids, knn_cls_ids, lim=True, limVal=10):
    print('')
    print('  | TestX ID | KNN IDs (Class IDs) - trainXY                        |')
    print('  |__________|______________________________________________________|')
    rows, cols = knn_ids.shape
    if lim and (rows>limVal): rows=limVal
    for i in range(0, rows):
      string = ''
      for j in range(0, cols):
        nstring = '{:4d}({:1d}) '.format(knn_ids[i,j], knn_cls_ids[i,j])
        string = string + nstring
      print('    {:4d}     | {}   '.format(i+1 , string))
    if lim: print('     ...     |  ...   ')
    print()
    return

  def print_est(self, testX, testY, Yest, vmode, acc, lim=True, limVal=10, test=''):
    if show_knn_details == True:
      print('---> Printing test results, in {} mode...'.format(vmode))
      print()
      print('  | TestX ID |   Y    |  Yest  |')
      print('  |__________|________|________|')
      rows =testX.shape[0]
      if lim and (rows>limVal): rows=limVal
      for x in range(0, rows):
        print('    {:4d}     | {:4d}   | {:4d} '.format(x+1 , testY[x].astype(int), Yest[x]))
      if lim: print('      ...    |   ...  |   ... ')
      print()
    if run_in_testmode:
      print('{} data ({}): {:.3f}% '.format(test, vmode,(acc*100)))
    else:
      print('   \-------->>> Prediction accuracy ({}): {:.3f}%  <<-|'.format(vmode,(acc*100)))
      print('\n')
    return
  # end of knn_classifier class ---------------------






if __name__ == '__main__':
  ''' ToDo: add a auto downloader
  '''

  ''' Image PCA and KNN
    dataset: https://www.kaggle.com/alessiocorrado99/animals10/download
    YouTube: https://www.youtube.com/watch?v=9YOWgQ4kHGg
    Image CPA from scratch:
    https://drscotthawley.github.io/blog/2019/12/21/PCA-From-Scratch.html
    https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
    CPA from scratch: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1

    read on shuffling error.
    https://stackoverflow.com/questions/39919181/indexerror-index-is-out-of-bounds-for-axis-0-with-size
  '''

  ''' Default config. Do not change.
  '''
  skip_process_raw_data = False # Do not change.
  skip_data_load = False # Do not change.
  skip_process_dataXY =  False # Do not change.
  skip_to_eigenXY = False # Do not change.
  prt = False # Do not change.
  plot_n_save = False
  show_knn_details = False
  run_in_testmode = False


  ''' NBUG config
  '''
  #skip_data_load = True # just to save time
  #skip_process_raw_data = True # set to False to save time, set True for first use.
  #skip_process_dataXY = True # just to save dev time
  #skip_to_eigenXY = True # used saved PCA dataset
  #plot_n_save = True
  #show_knn_details = True
  run_in_testmode = True

  ''' run config
  '''
  #prt = True
  knn_k = 7
  image_size = 48 # pixels, equal height and width
  samples_per_class = 300
  shuffles = 10 # shuffle n times to mix data
  testfrac = .2 # 0-1.0 | test set fraction of main set
  PCr = 100 # % percentage of principal components used
  numPC = int((PCr/100)*image_size**2)
  clen = len(animals)
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
    #keysXY = np.full((0, mfeat+ 1), 1)
    #keyy = np.full((1, mfeat+ 1), 1)
    labels = list()
    eigenXY_class = np.empty((samples_per_class, mfeat+ 1))
    eigenXY = np.empty((0, numPC+ 1))
    print()
    # get data per class and process to obtain PCA
    for b, (directory, label) in enumerate(translations.items()):
      if label in animals:
        X, Y = get_data(data, label=label)
        V,S,img_mean, Z = pca(X, numPC)
        #set_trace()
        dataXY.append([label,X,Y,V,S,img_mean])
        labels.append(label)
        #key = np.reshape(img_mean, (1,-1))
        # copy processed PCA keys - mean
        #set_trace()
        Y = np.expand_dims(Y,axis=1)
        # new data set
        #print('   |---> Y.shape: ', Y.shape)
        #print('   |---> Z.shape: ', Z.shape)
        eigenXY_class = copy.deepcopy(np.concatenate((Z,Y), axis=1))
        eigenXY = copy.deepcopy(np.concatenate((eigenXY,eigenXY_class), axis=0))
        #print(eigenXY)
        #set_trace()
        # plot or save images
        #print('   |---> V.shape: ', V.shape)
        eImgs = Z.dot(V.T)
        #print('   |---> eImgs.shape: ', eImgs.shape)
        if plot_n_save:
          plot_utility(label, image_size, eImgs, img_mean, b, PCr)
          plot_PCvariance(label, image_size, b, PCr)

    # save processed dataset object
    print('---> Saving dataXY object to binary file...')
    with open('dataXY.json', 'wb') as dataXY_file:
      pkl.dump(dataXY, dataXY_file)

    # save keysXY dataset object
    #print('---> Saving keysXY object to binary file...')
    #with open('keysXY.json', 'wb') as keysXY_file:
      #pkl.dump(keysXY, keysXY_file)

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
      #print('---> Loading keysXY object from binary file...')
      #with open('keysXY.json', 'rb') as keysXY_file:
        #keysXY = pkl.load(keysXY_file)
    else:
      # load eigenXY dataset object
      print('---> Loading eigenXY object from binary file...')
      with open('eigenXY.json', 'rb') as eigenXY_file:
        eigenXY = pkl.load(eigenXY_file)

      # load labels dataset object
      print('---> Loading labels object from binary file...')
      with open('labels.json', 'rb') as labels_file:
        labels = pkl.load(labels_file)


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

  knn = knn_classifier(trainXY, testXY, k=knn_k, prt=True, vmode='weighted', labels=labels)

  print('------------//>>')
  print()
  print('Experiment tag: {}'.format(get_tag(clen, '', '', image_size, PCr)))
  #print('\------> Test model with: training data')
  knn.get_acc(knn.trainX, knn.trainY, vmode='weighted', test='training')
  knn.get_acc(knn.trainX, knn.trainY, vmode='simple', test='training')

  #print('\------> Test model with: test data')
  knn.get_acc(knn.testX, knn.testY, vmode='weighted', test='test')
  knn.get_acc(knn.testX, knn.testY, vmode='simple', test='test')


  print('---> End of process.')


























































































# EOF
