from PIL import Image
import numpy as np
import matplotlib as plt
import os, shutil

# acceptable image formats
exts = {'.png', '.jpg', '.jpeg'}


translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", \
  "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", \
  "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", \
  "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", \
  "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", \
  "squirrel": "scoiattolo"}


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
        pixels = pixels/255.0
        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('mean: %.3f, std: %.3f' % (mean, std))
        mean, std = pixels.mean(), pixels.std()
        if self.prt is True:
          print('new --> mean: %.3f, std: %.3f' % (mean, std))
          print('    --> min:  %.3f, max: %.3f' % (pixels.min(), pixels.max()))
        image_n = Image.fromarray(pixels, mode='RGB')
        image_n.save(self.dest+fname+ext)
        #image_n.show()
    return
#end of class std_image:

def create_data_obj(src, size, categories):
  data = list()


  return data


if __name__ == '__main__':

  ''' ToDo: add a auto downloader
  '''
  #wget https://www.kaggle.com/alessiocorrado99/animals10/download
  stdImages = std_image(src='./data/raw-img/cane/', dest='./data/raw-img/cane/n_cane/')
  stdImages.norm_img()

  # create data object
  data = create_data_obj(src='./data/temp_data/', size=100, categories=5)
