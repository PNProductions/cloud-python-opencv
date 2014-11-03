#!/usr/bin/env python

import sys
import os
from os.path import basename, splitext, isfile
import glob
import numpy as np
import time
import cv2
from scipy.io import savemat, loadmat
from videoseam import seam_merging, progress_bar
from tvd import TotalVariationDenoising

sys.path.insert(0, './utils')
from image_helper import image_open, local_path, to_matlab_ycbcr
from video_helper import save_video_caps
from parallelize import parallelize

progress_bar(False)

def generate_step(I, img):
  result = np.empty((img.shape[0], img.shape[1], img.shape[2]))
  r = np.arange(img.shape[2])
  for i in xrange(img.shape[0]):
    result[i] = r > np.vstack(I[i])
  return result.astype(np.uint64)

def print_seams(result, seams):
  print "printing seam"
  seams = seams.astype(np.uint64)
  A = np.zeros_like(result)  # np.ones_like(result) * 255
  correction = np.zeros((result.shape[0], result.shape[1], result.shape[2])).astype(np.uint64)
  for i in xrange(seams.shape[0]):
    X, Y = np.mgrid[:result.shape[0], :result.shape[1]]
    I = seams[i]
    I = I + correction[X, Y, I]
    color = np.random.rand(3) * 255
    A[X, Y, I] = color
    correction = correction + generate_step(I, result)
  return A

deleteNumberW = 1
counting_frames = 10
path = './testing_videos/4_videos/*'

for i in xrange(len(sys.argv) - 1):
  if sys.argv[i] == '-s':
    deleteNumberW = int(sys.argv[i + 1])
  elif sys.argv[i] == '-f':
    counting_frames = int(sys.argv[i + 1])
  # elif sys.argv[i] == '-p':
  #   path = str(sys.argv[i + 1])

onlyfiles = glob.glob(path)
suffix = ''

alpha = 0.5
betaEn = 0.5

iterTV = 80
makeNewDecData = False

debug = False
saveBMP = True

def batch_videos(filename):
  cap = cv2.VideoCapture(filename)
  size = '_reduce' if deleteNumberW < 0 else '_enlarge'
  size += str(-deleteNumberW) if deleteNumberW < 0 else str(deleteNumberW)

  name = splitext(basename(filename))[0] + suffix + '_' + size + '_' + str(int(time.time()))

  frames_count, fps, width, height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), cap.get(cv2.cv.CV_CAP_PROP_FPS), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

  frames_count = frames_count if counting_frames is None else counting_frames

  importance = np.empty((frames_count, height, width))
  structureImage = np.empty((frames_count, height, width))
  video = np.empty((frames_count, height, width, 3))

  mat_name = './temp/' + splitext(basename(filename))[0] + '_' + str(frames_count) + '_garbage.mat'

  if isfile(mat_name):
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
  A = np.clip(A * 0.8 + video, 0, 255).astype(np.uint8)

  result = np.clip(result, 0, 255).astype(np.uint8)
  save_video_caps(result, './results/' + name + '_')
  save_video_caps(A, './results/' + name + '_seams_')
  cap.release()
  print 'Finished file: ' + basename(filename)


methods_batch = ([], [])
for filename in onlyfiles:
  methods_batch[0].append(batch_videos)
  methods_batch[1].append((filename,))

parallelize(methods_batch[0], methods_batch[1])
