#!/usr/bin/python
import sys
import os
from os.path import basename, splitext, isfile
import glob
from numpy import size, concatenate
import numpy as np
import time
import cv2
from scipy.io import savemat, loadmat

def generate_step(I, img):
  result = np.empty((img.shape[0], img.shape[1], img.shape[2]))
  r = np.arange(img.shape[2])
  for i in xrange(img.shape[0]):
    result[i] = r > np.vstack(I[i])
  print result
  return result.astype(np.uint64)

def print_seams(result, seams):
  print "printing seam"
  seams = seams.astype(np.uint64)
  A = np.ones_like(result) * 255
  correction = np.zeros((result.shape[0], result.shape[1], result.shape[2])).astype(np.uint64)
  print seams.shape[0]
  for i in xrange(seams.shape[0]):
    X, Y = np.mgrid[:result.shape[0], :result.shape[1]]
    I = seams[i]
    I = I + correction[X, Y, I]
    color = np.random.rand(3) * 255
    A[X, Y, I] = color
    correction = correction + generate_step(I, result)
  return A


if len(sys.argv) >= 2 and sys.argv[1] is not None:
  deleteNumberW = int(sys.argv[1])
else:
  deleteNumberW = 10

deleteNumberH = 0

suffix = ''
sys.path.insert(0, 'py-video-retargeting/src')


from image_helper import image_open, local_path, to_matlab_ycbcr
from seam_merging import seam_merging
from tvd import TotalVariationDenoising
from video_helper import save_video_caps

# path = './testing_images/reduction/*' if deleteNumberW < 0 else './testing_images/enlargement/*'
path = './testing_videos/*'

onlyfiles = glob.glob(path)

alpha = 0.5
betaEn = 0.5

iterTV = 80
makeNewDecData = False

debug = False
saveBMP = True

for filename in onlyfiles:
  print 'Opening file ' + basename(filename)
  cap = cv2.VideoCapture(filename)
  size = '_reduce' if deleteNumberW < 0 else '_enlarge'
  size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)

  name = splitext(basename(filename))[0] + suffix + '_' + '_' + size + '_' + str(int(time.time()))

  frames_count, fps, width, height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), cap.get(cv2.cv.CV_CAP_PROP_FPS), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

  print frames_count, fps, width, height

  frames_count = 10

  importance = np.empty((frames_count, height, width))
  structureImage = np.empty((frames_count, height, width))
  video = np.empty((frames_count, height, width, 3))

  mat_name = splitext(basename(filename))[0] + '_garbage.mat'

  if isfile(splitext(basename(filename))[0] + '_garbage.mat'):
    r = loadmat(mat_name)
    video, structureImage, importance = r['video'], r['structureImage'], r['importance']
  else:
    i = 0
    while cap.isOpened() and i < frames_count:
      ret, X = cap.read()
      if not ret:
        break
      y = cv2.cvtColor(X, cv2.COLOR_BGR2YCR_CB)
      y = y.astype(np.float64)
      structureImage[i] = TotalVariationDenoising(y[:, :, 0], iterTV).generate()

      kernel = np.array([[0, 0, 0],
                         [1, 0, -1],
                         [0, 0, 0]
                         ])

      importance[i] = np.abs(cv2.filter2D(y[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)) + np.abs(cv2.filter2D(y[:, :, 0], -1, kernel.T, borderType=cv2.BORDER_REPLICATE))

      video[i] = X
      i += 1

    savemat(mat_name, { 'importance' : importance, 'structureImage' : structureImage, 'video' : video }, do_compression=True)

  result, seams = seam_merging(video, structureImage, importance, deleteNumberW, alpha, betaEn)

  A = print_seams(video, seams)
  A = np.clip(A, 0, 255).astype(np.uint8)

  result = np.clip(result, 0, 255).astype(np.uint8)
  # Saving the result:
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  # out = cv2.VideoWriter('./results/' + name + '.avi', fourcc, fps, (result.shape[2], result.shape[1]))
  # for i in xrange(result.shape[0]):
  #   out.write(result[i])

  save_video_caps(result, './results/' + name + '_')
  save_video_caps(A, './results/' + name + '_seams_')

cap.release()
#out.release()
