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

'''
class std_image:
  def __init__(self, src='./', dest='./out/', prt_setting=True):
    self.src = src
    self.dest = dest
    self.prt = prt_setting
    #end of def __init__

  def check_dest(self):
    try:
      os.makedirs(self.dest, exist_ok=True)
      print('--> output directory %s successfully created.' % self.dest)
    except OSError as error: # if returns error
      print('--> output directory %s already exists.' % self.dest)
      shutil.rmtree(self.dest)
      print('--> output directory %s is removed.' % self.dest)
      os.makedirs(self.dest)
      print('--> output directory %s successfully created.' % self.dest)
    return

  def gen_pos_global_std(self, dest='pg_std/'):
    self.dest = self.src+dest
    self.check_dest()
    for i, file in enumerate(os.listdir(self.src)):
      fname, ext = os.path.splitext(file)
      if ext in exts:
        if self.prt is True:
          print()
          print('- %5d  %s%s' % (i,fname,ext))
        image = Image.open(self.src+fname+ext)
        pixels = np.asarray(image)
        pixels = pixels.astype('float32')
        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('mean: %.3f, std: %.3f' % (mean, std))
        pixels = (pixels - mean) /std # get global std of pixels
        pixels = np.clip(pixels, -1.0, 1.0) # clip to [-1,1]
        pixels = (pixels + 1.0) / 2.0 # shift pixels to have mean at .5
        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('new --> mean: %.3f, std: %.3f' % (mean, std))
          print('    --> min:  %.3f, max: %.3f' % (pixels.min(), pixels.max()))
        image_n = Image.fromarray(pixels, mode='RGB')
        image_n.save(self.dest+fname+ext)
        #image_n.show()
    return

  def norm_img(self, dest='norm/'):
    self.dest = self.src+dest
    self.check_dest()
    for i, file in enumerate(os.listdir(self.src)):
      fname, ext = os.path.splitext(file)
      if ext in exts:
        if self.prt is True:
          print()
          print('- %5d  %s%s' % (i,fname,ext))
        image = Image.open(self.src+fname+ext)
        pixels = np.asarray(image)
        pixels = pixels.astype('float32')

        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('mean: %.3f, std: %.3f' % (mean, std))
        pixels = pixels/255.0
        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('new --> mean: %.3f, std: %.3f' % (mean, std))
          print('    --> min:  %.3f, max: %.3f' % (pixels.min(), pixels.max()))
        image_n = Image.fromarray(pixels, mode='RGB')
        image_n.save(self.dest+fname+ext)
        #image_n.show()
    return
#end of class std_image:
'''

def create_data_obj(src, subdir, categories, n_samples, img_size=250): # size, categories):
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
    print('---> Saved image: '+figname)

  #fig.show(bbox_inches='tight',dpi=100)
  return


class KNN:
  def __init__(self, trainXY, testX=None, k=1, precision=4):
    self.precision = precision

    set_trace()

    self.nFold_CrossValidation(trainXY, nfolds=10, k=k, prt=False)
    print('\n')
    if testX is not None:
      print('test dataset:')
      print(testX)
      self.predict_testset(trainXY, testX, k, precision, prt=True)
    print("--- end of KNN process --- \n\n\n")
    return

  def predict_testset(self, trainXY, testX, k, precision=4, prt=True):
      self.prt = prt
      self.precision = precision
      self.predictions = list()
      for datum in testX:
          prediction = self.predict_class(trainXY, datum, k)
          self.predictions.append(prediction)
      if self.prt == True:
          print()
          print("Summary:")
          print('k: ', k)
          print('Test set:', testX)
          print('Predictions:  ', self.predictions)
          print('Precision: ', precision, " sigfig")
      return self.predictions

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

  # Calculate nearest neighbors
  def get_neighbors(self, trainX, test_row, k): # trainX, trainX_i, # of nearest neighbors
      distances = list()
      neighbors = list()
      if self.prt == True:
          print()
          print()
          print('Calculate distances to datum: ', test_row)
          print('Training data XY  |   Distance |')
          print('__________________|____________|')
      for X_i in trainX:
          dist = self.euclidean_distance(X_i, test_row)
          if self.prt == True:
                  print(X_i, ' |  ', dist)
          distances.append((X_i, dist))
      distances.sort(key=lambda tup: tup[1])
      for i in range(k): neighbors.append(distances[i][:])
      #self.print_distances(distances)
      #self.print_neighbors(neighbors)
      if self.prt == True:
          print()
          print(k, ' Nearest Neighbors:')
          for i in neighbors:
              print(i)
      return neighbors

  def predict_class(self, trainXY, testX, k):
      #print(trainXY.dtype)
      #print(testX.dtype)
      #self.trainXY = np.asarray([sublist for sublist in trainXY], dtype=object)
      neighbors = self.get_neighbors(trainXY, testX, k)
      output_values = [row[-2][1] for row in neighbors]
      if self.prt == True:
          print()
          print('KNN classes: ', output_values)
      self.prediction = max(set(output_values), key=output_values.count)
      if self.prt == True:
          print('Datum prediction: ', self.prediction)
      return (self.prediction)

  def nFold_CrossValidation(self, trainXY, nfolds, k, prt=False):
      #print("trainXY", trainXY)
      self.scores = list()
      XYfolds = self.split_trainXY(trainXY, nfolds)
      for fold in XYfolds:
          testFold = list()
          trainFolds = list(XYfolds)
          trainFolds.remove(fold)
          trainFolds = sum(trainFolds, [])
          for datum in fold:
              datum_copy = list(datum)
              testFold.append(datum_copy[0])
              datum_copy[-1] = None
              #print("nfold - test datum:", datum_copy)
          prediction = self.predict_testset(trainFolds, testFold, k, prt=False)
          groundtruth = [datum[-1] for datum in fold]
          acc = self.get_acc(groundtruth, prediction)
          self.scores.append(acc)
      print()
      print("Model accuracy is based on training data and ", nfolds, " folds cross validations.")
      print("acc: ", self.scores)
      overall_acc = (sum(self.scores)/float(len(self.scores)))
      overall_acc = round(overall_acc, 2)
      print("Overall model accuracy: ", overall_acc,'%')
      return

  # split training dataset into n-folds for cross validation
  def split_trainXY(self, trainXY, nfolds):
      trainXY_nfolded = list()
      trainXY_list = list(trainXY)
      fold_size = len(trainXY)/nfolds
      for i in range(nfolds):
          new_fold = list()
          while len(new_fold) < fold_size:
              idx = rand.randrange(len(trainXY_list))
              new_fold.append(trainXY_list.pop(idx))
          trainXY_nfolded.append(new_fold)
      return trainXY_nfolded

  def get_acc(self, groundtruth, prediction):
      correct = 0
      for i in range(len(groundtruth)):
          if groundtruth[i] == prediction[i]:
              correct += 1
      return correct / float(len(groundtruth)) * 100.0








if __name__ == '__main__':

  ''' ToDo: add a auto downloader
  '''
  #url = 'https://www.kaggle.com/alessiocorrado99/animals10/download'

  ''' Image PCA
    YouTube: https://www.youtube.com/watch?v=9YOWgQ4kHGg
    Image CPA from scratch:
    https://drscotthawley.github.io/blog/2019/12/21/PCA-From-Scratch.html
    https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
    CPA from scratch: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

  '''

  ''' Default config. Do not change.
  '''
  skip_process_raw_data = False # Do not change.
  skip_data_load = False # Do not change.
  skip_process_dataXY =  False # Do not change.
  skip_to_eigenXY = False

  ''' NBUG config
  '''
  skip_data_load = True # just to save time
  skip_process_raw_data = True # set to False to save time, set True for first use.
  skip_process_dataXY = True # just to save dev time
  skip_to_eigenXY = True # used saved PCA dataset

  ''' run config
  '''
  prt = True
  image_size = 100 # pixels, equal height and width
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

    # print labels
    print()
    print('---> labels:')
    pp(labels)
    print()

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

  ''' keysXY: set of keys for all known classes
  '''
  #print()
  #print('Class keys (mean projection):')
  #pp(keysXY)
  print()
  print('Processed dataset (PCA):')
  pp(eigenXY)
  print()
  print('Shuffling processed dataset (PCA):')
  set_trace() # ----------------------------------------------------------------------------->>>>>>

  for _ in range(shuffles):
    eigenXY = np.random.shuffle(eigenXY)
  print()
  print('Processed dataset (PCA): please inspect...')
  pp(eigenXY)

  set_trace()
  nSamps = eigenXY.shape[0]

  test_size = int(testfrac * nSamps)

  testXY = eigenXY[np.random.choice(nSamps, size=test_size, replace=False)]
  testX = copy.deepcopy(testXY[:][:-1])
  testX = copy.deepcopy(testXY[:][:-1])

  knn_classifier = KNN(eigenXY, testX, testY, k=5)




























































































# EOF
