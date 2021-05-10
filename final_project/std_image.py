from PIL import Image, ImageOps
import numpy as np
import matplotlib as plt
import os, shutil
from random import randrange


# acceptable image formats
exts = {'.png', '.jpg', '.jpeg'}

prt = True
translations = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel"}


animals=['dog', 'horse','elephant', 'butterfly', 'chicken', 'cat', 'cow', \
 'sheep', 'squirrel','spider']

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




def create_data_obj(src, subdir, categories, n_samples): # size, categories):
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
          image = change_contrast(image, 100) # change contrast for grey scale
          image = ImageOps.grayscale(image)
          pixels = np.asarray(image)
          pixels = pixels.astype('float32')
          data.append([i, pixels, target, src, category,  fname+ext])
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
  factor = (259 * (level + 255))/(255 * (259 - level))
  def contrast (c):
    return 128 + factor * (c-128)
  return image.point(contrast)



# split training dataset into n-folds for cross validation
def split_trainXY(trainXY, nfolds):
  trainXY_nfolded = list()
  trainXY_list = list(trainXY)
  fold_size = len(trainXY)/nfolds
  for i in range(nfolds):
    new_fold = list()
    while len(new_fold) < fold_size:
      idx = randrange(len(trainXY_list))
      new_fold.append(trainXY_list.pop(idx))
    trainXY_nfolded.append(new_fold)
  return trainXY_nfolded

def get_acc(groundtruth, prediction):
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
    https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
  '''

  # create data object
  data = create_data_obj(src='./data/', subdir='raw-img/', \
    categories=translations, n_samples=2)

  data = norm_images(data, save=True) # normalize dataset

  #random.shuffle(data)

  X = list()
  Y = list()
  for datum in data:
    X.append(datum[1])
    Y.append(datum[2])























































































# EOF
