import torch
import numpy
from sklearn.metrics import roc_curve, auc 


def one_hot(labels, num_classes, device = None, dtype  = None, eps= 1e-6):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def binary_mapping(x, threshold=0.62):
  return numpy.array((x > threshold).astype(int))

def accuracy(output, target):
  output = binary_mapping(output)
  target = binary_mapping(target)
  TP = numpy.sum(numpy.logical_and(output == 1, target == 1))
  TN = numpy.sum(numpy.logical_and(output  == 0, target == 0))
  FP = numpy.sum(numpy.logical_and(output  == 1, target == 0))
  FN = numpy.sum(numpy.logical_and(output == 0, target == 1))
  return (TP + TN)/ (TP + FP + TN + FN)

def SE(output, target):
  output = binary_mapping(output)
  target = binary_mapping(target)
  TP = numpy.sum(numpy.logical_and(output == 1, target == 1))
  FN = numpy.sum(numpy.logical_and(output == 0, target == 1))
  return TP/(TP + FN)

def SP(output, target):
  output = binary_mapping(output)
  target = binary_mapping(target)
  TN = numpy.sum(numpy.logical_and(output  == 0, target == 0))
  TP = numpy.sum(numpy.logical_and(output == 1, target == 1))
  FP = numpy.sum(numpy.logical_and(output  == 1, target == 0))
  return TN/ (TN + FP)


def AUC(output, target):
  output = binary_mapping(output).ravel()
  target = binary_mapping(target).ravel()

  fpr, tpr, _ = roc_curve(target, output)
  return auc(fpr,tpr)

