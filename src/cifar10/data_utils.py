import os
import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf

num_classes = 10
def flip_labels_C(corruption_prob):
  '''
  returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
  concentrated in only one other entry for each row
  '''
  np.random.seed(1)

  C = np.eye(num_classes) * (1 - corruption_prob)
  row_indices = np.arange(num_classes)
  for i in range(num_classes):
    C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
  return C

def uniform_mix_C(mixing_ratio):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1.0 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)

def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print file_name
    full_name = os.path.join(data_path, file_name)
    with open(full_name) as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print "Prepropcess: [subtract mean], [divide std]"
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print "mean: {}".format(np.reshape(mean * 255.0, [-1]))
  print "std: {}".format(np.reshape(std * 255.0, [-1]))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

def read_data_corrupt_label(data_path, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  # Corrupt the labels
  gold_fraction = 0.05
  cifar_labels = labels["train"]
  num_gold = int(len(cifar_labels) * gold_fraction)
  num_silver = len(cifar_labels) - num_gold
  num_classes = 10
  corruption_matrix = uniform_mix_C(0.8)
  print('corruption matrix:\n{}'.format(corruption_matrix))

  for i in range(num_silver):
    cifar_labels[i] = np.random.choice(num_classes, p=corruption_matrix[cifar_labels[i]])

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print "Prepropcess: [subtract mean], [divide std]"
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print "mean: {}".format(np.reshape(mean * 255.0, [-1]))
  print "std: {}".format(np.reshape(std * 255.0, [-1]))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels