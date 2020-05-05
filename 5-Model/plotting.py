import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import sys

# load data
f_s = h5py.File("/home/jua/doublehiggs_jetimage/4-Dataset/dataset_hdf5/{}_dataset.hdf5".format(sys.argv[1]),"r")
f_b = h5py.File("/home/jua/doublehiggs_jetimage/4-Dataset/dataset_hdf5/{}_dataset.hdf5".format(sys.argv[2]),"r")

def make_plot_n(n):
  s = f_s['train_images'][:][n]
  b = f_b['train_images'][:][n]

  s1 = s[0]+s[1]+s[2]+s[3]+s[4]
  b1 = b[0]+b[1]+b[2]+b[3]+b[4]
  '''
  fig, (ax0,ax1) = plt.subplots(1,2)
  ax0.set_title('signal b jet')
  ax1.set_title('background b jet')

  im0 = ax0.imshow(s1)
  plt.colorbar(im0)
  im1 = ax1.imshow(b1)
  plt.colorbar(im1)

  #plt.imshow(s1)
  #plt.imshow(b1)
  #plt.colorbar()
  #ax0.colorbar()
  plt.show()
  '''
  plt.subplot(211)
  plt.imshow(s1)
  plt.subplot(212)
  plt.imshow(b1)
  plt.subplots_adjust(bottom = 0.1, right=0.8,top=0.9)
  cax = plt.axes([0.85, 0.1 ,0.075,0.8])
  plt.colorbar(cax=cax)
  plt.show()
make_plot_n(0)
