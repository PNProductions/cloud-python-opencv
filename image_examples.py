import sys
import os
from os.path import basename, splitext
import glob
from numpy import size, concatenate
import numpy
import time
import cv2

print sys.argv[0]

if len(sys.argv) >= 2 and sys.argv[1] is not None:
  deleteNumberW = int(sys.argv[1])
else:
  deleteNumberW = -100

deleteNumberH = 0

suffix = None
if len(sys.argv) >= 3 and sys.argv[2] == 'original':
  suffix = '_original'
  sys.path.insert(0, 'py-seam-merging/src')
else:
  suffix = '_modified'
  sys.path.insert(0, 'modified-video-seam-merging/src')



from image_helper import image_open, image_save, local_path, to_matlab_ycbcr
from seam_merging import seam_merging
from tvd import TotalVariationDenoising

path = './testing_images/reduction/*' if deleteNumberW < 0 else './testing_images/enlargement/*'
onlyfiles = glob.glob(path)

alpha = 0.5
betaEn = 0.5

iterTV = 80
makeNewDecData = False

debug = False
saveBMP = True

for filename in onlyfiles:
  print 'Opening file ' + basename(filename)
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