#!/usr/bin/env python

import sys
import os
from os.path import basename, splitext
import glob
from numpy import size, concatenate
import numpy
import time
import cv2
from tvd import TotalVariationDenoising
from videoseam import seam_merging, progress_bar

sys.path.insert(0, './utils')
from parallelize import parallelize
from image_helper import image_open, image_save

progress_bar(False)

if len(sys.argv) >= 2 and sys.argv[1] is not None:
  deleteNumberW = int(sys.argv[1])
else:
  deleteNumberW = -100

deleteNumberH = 0

suffix = None
if len(sys.argv) >= 3 and sys.argv[2] == 'original':
  suffix = '_original'
  # sys.path.insert(0, 'py-seam-merging/examples')
else:
  suffix = '_modified'
  # sys.path.insert(0, 'py-video-retargeting/examples')


path = './testing_images/reduction/*' if deleteNumberW < 0 else './testing_images/enlargement/*'
onlyfiles = glob.glob(path)

alpha = 0.5
betaEn = 0.5

iterTV = 80
makeNewDecData = False

debug = False
saveBMP = True

def batch_images(filename):
  X = image_open(filename)
  y = cv2.cvtColor(X, cv2.COLOR_BGR2YCR_CB)
  y = y.astype(numpy.float64)
  structureImage = TotalVariationDenoising(y[:, :, 0], iterTV).generate()

  importance = y
  kernel = numpy.array([[0,  0,  0],
                        [1,  0, -1],
                        [0,  0,  0]
                        ])
  importance = numpy.abs(cv2.filter2D(y[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)) + numpy.abs(cv2.filter2D(y[:, :, 0], -1, kernel.T, borderType=cv2.BORDER_REPLICATE))

  img = seam_merging(X, structureImage, importance, deleteNumberW, alpha, betaEn)
  size = '_reduce' if deleteNumberW < 0 else '_enlarge'
  size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)
  
  name = splitext(basename(filename))[0] + suffix + '_' + '_' + size + '_' + str(int(time.time()))
  image_save(img, name, './results/')
  print filename + ' finished!'

methods_batch = ([], [])
for filename in onlyfiles:
  methods_batch[0].append(batch_images)
  methods_batch[1].append((filename,))


parallelize(methods_batch[0], methods_batch[1])
