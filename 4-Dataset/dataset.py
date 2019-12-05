import os, sys
from ROOT import *
from array import array
import numpy as np
import glob
from pprint import pprint

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

class DoubleHiggsDataset(Sequence):
  def __init__(self, folder_name, path, batch_size, max_len):
    self.data = folder_name
    self.root_file = TFile(path)
    self.tree = self.root_file.events
    self.num_entries = self.tree.GetEntries()
    self.batch_size = batch_size
    self.max_len = max_len
    

    self.x_names = [
      'image_chad_pt_33_1',
      'image_nhad_pt_33_1',
      'image_electron_pt_33_1',
      'image_muon_pt_33_1',
      'image_photon_pt_33_1',
   
      'image_chad_pt_33_2',
      'image_nhad_pt_33_2',
      'image_electron_pt_33_2',
      'image_muon_pt_33_2',
      'image_photon_pt_33_2',
    
      'image_chad_mult_33_1',
      'image_nhad_mult_33_1',
      'image_electron_mult_33_1',
      'image_muon_mult_33_1',
      'image_photon_mult_33_1',
    
      'image_chad_mult_33_2',
      'image_nhad_mult_33_2',
      'image_electron_mult_33_2',
      'image_muon_mult_33_2',
      'image_photon_mult_33_2'
    ]

  def __len__(self):
    return int(self.num_entries / float(self.batch_size))

  def __getitem__(self, index):
    start = index * self.batch_size
    end = (index+1) * self.batch_size
    x = []
    y = []
    for entry in range(start, end):
      self.tree.GetEntry(entry)

      # Set x
      x_array = [np.array(getattr(self.tree, each), dtype=np.float32) for each in self.x_names]
      
      # Set y
      if self.data == "pp_hh":
        y_array = np.ones((len(x_array)), dtype=np.int64)
      else :
        y_array = np.zeros((len(x_array)), dtype=np.int64)
      x.append(x_array)
      y.append(y_array)
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_len,padding='post',truncating='post', dtype= np.float32)
    y = keras.preprocessing.sequence.pad_sequences(y, maxlen=self.max_len,padding='post',truncating='post', dtype= np.int32)
    return x, y

def get_datasets(folder_name, folder_path, batch_size, max_len):
  dataset = glob.glob(folder_path+"*.root")
  print(dataset)
  datasets = [
    DoubleHiggsDataset(folder_name, dataset[0], batch_size, max_len),
    DoubleHiggsDataset(folder_name, dataset[1], batch_size, max_len),
    DoubleHiggsDataset(folder_name, dataset[2], batch_size, max_len)
  ]
  train_set, val_set, test_set = sorted(datasets, key=lambda dset: len(dset), reverse=True)
  return train_set, val_set, test_set

def main():
  batch_size = 10
  max_len = 20
  folder_name = sys.argv[1]
  folder_path = './split/{}/'.format(folder_name)
  print(folder_name, folder_path)
  train_set, val_set, test_set = get_datasets(folder_name,folder_path,batch_size, max_len)
  print("Train Set : ",train_set, len(train_set) )
  print("Val Set : ",val_set, len(val_set) )
  print("Test Set : ",test_set, len(test_set) )
  print(train_set[50])

if __name__ == '__main__':
  main()
