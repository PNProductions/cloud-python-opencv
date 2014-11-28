#!/usr/bin/env python

import sys
import argparse
from os.path import basename, splitext, isfile
import glob
import numpy as np
import cv2
from scipy.io import savemat, loadmat
import videoseam as vs
import seamcarving as sc
from utils.optical_flow import farneback
from utils.video_helper import save_video_caps
from utils.parallelize import parallelize
from tvd import TotalVariationDenoising

vs.PROGRESS_BAR = True
suffix = ''

# -------------------------------------------------------
# Algorithm constraint
# -------------------------------------------------------
alpha = 0.5
beta = 0.5
iterTV = 80
# -------------------------------------------------------

makeNewDecData = False
debug = False
saveBMP = True

# -------------------------------------------------------
# Argument parser parameters
# -------------------------------------------------------

# seam = 1
# frame = 10
# save_importance = 0
# allowed_methods = ["seam_merging", "seam_carving", "time_merging"]
# method = allowed_methods[0]
# path = 'testing_videos/downsample/japan/'
# save_vectors = True
# global_vector = False

parser = argparse.ArgumentParser(description='Video retargeting with different methods')
parser.add_argument('-p', '--path', default='testing_videos/downsample/japan/', type=str, help='Path for the videos to retarget')
parser.add_argument('-s', '--seam', default=1, type=int, help='Seam to use (default %(default)s)')
parser.add_argument('-f', '--frame', type=int, help='Frame to use (default all video frame)')
parser.add_argument('-i', '--save-importance', action='store_true', help='Save importance map')
parser.add_argument('-v', '--save-vectors', action='store_true', dest='save_vectors', help='Save motion vector')
parser.add_argument('-nv', '--no-save-vectors', action='store_false', dest='save_vectors', help='Save motion vector')
parser.add_argument('-g', '--global-vector', action='store_true', help='Use global motion vector')
parser.add_argument('-m', '--method', default='seam_merging', choices=['seam_merging', 'seam_carving', 'time_merging'], help='Algorithm to use')
parser.set_defaults(save_vectors=True)
args = parser.parse_args()

# -------------------------------------------------------
# -------------------------------------------------------
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


def batch_videos(filename):
  cap = cv2.VideoCapture(filename)
  size = '_reduce' if args.seam > 0 else '_enlarge'
  size += str(-args.seam) if args.seam < 0 else str(args.seam)

  name = splitext(basename(filename))[0] + suffix + '_' + size + '_' + args.method

  frame_count, fps, width, height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), cap.get(cv2.cv.CV_CAP_PROP_FPS), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

  frame_count = frame_count if args.frame is None else args.frame

  importance = np.empty((frame_count, height, width))
  cartoon = np.empty((frame_count, height, width))
  video = np.empty((frame_count, height, width, 3))
  vectors = np.empty((frame_count, height, width))

  mat_name = './temp/' + splitext(basename(filename))[0] + '_' + str(frame_count) + '_garbage.mat'

  if isfile(mat_name):
    r = loadmat(mat_name)
    video, cartoon, importance = r['video'], r['cartoon'], r['importance']
  else:
    i = 0
    while cap.isOpened() and i < frame_count:
      ret, X = cap.read()
      if not ret:
        break
      y = cv2.cvtColor(X, cv2.COLOR_BGR2YCR_CB)
      y = y.astype(np.float64)
      cartoon[i] = TotalVariationDenoising(y[:, :, 0], iterTV).generate()

      # Motion vector frame per frame (filtered with medianBlur to delete salt and pepper noise)
      if i > 0:
        vectors[i - 1] = farneback(cartoon[i - 1], cartoon[i])
        vectors[i - 1] = cv2.medianBlur(vectors[i - 1].astype(np.uint8), 19).astype(np.float64)

      kernel = np.array([[0, 0, 0],
                         [1, 0, -1],
                         [0, 0, 0]
                         ])

      importance[i] = np.abs(cv2.filter2D(y[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)) + np.abs(cv2.filter2D(y[:, :, 0], -1, kernel.T, borderType=cv2.BORDER_REPLICATE))

      video[i] = X
      i += 1

    savemat(mat_name, {'importance': importance, 'cartoon': cartoon, 'video': video}, do_compression=True)

  # Motion vector normalizing and saving
  maxv = vectors.max(axis=0)
  maxv[np.where(maxv == 0)] = 1
  vectors = vectors[:] / maxv * 255
  # Use a global motion vector (consider every frame)
  if args.global_vector:
    normalized = vectors.sum(axis=0)
    normalized = np.clip(normalized, 0, 255)
    if args.save_vectors:
      vecs = normalized[:, :, np.newaxis]
      cv2.imwrite('./results/' + name + '_vectors.bmp', vecs[:, :, [0, 0, 0]])
    vectors[:] = normalized
  # Use a frame per frame motion vector
  else:
    if args.save_vectors:
      v2 = np.clip(vectors, 0, 255).astype(np.uint8)
      v2 = np.expand_dims(v2, axis=3)[:, :, :, [0, 0, 0]]
      save_video_caps(v2, './results/' + name + '_vectors_')

  if args.save_importance:
    imp = np.clip(importance, 0, 255).astype(np.uint8)
    imp = np.expand_dims(imp, axis=3)[:, :, :, [0, 0, 0]]
    save_video_caps(imp, './results/' + name + '_importance_')

  if args.method == "seam_merging":
    result, seams = vs.seam_merging(video, cartoon, importance, vectors, args.seam, alpha, beta)
  elif args.method == "seam_carving":
    result, seams = sc.seam_carving(video, args.seam)
  elif args.method == "time_merging":
    result, seams = vs.time_merging(video, cartoon, importance, None, args.seam, alpha, beta)

  # Saving seams frame images
  A = print_seams(video, seams)
  A = np.clip(A * 0.8 + video, 0, 255).astype(np.uint8)
  result = np.clip(result, 0, 255).astype(np.uint8)
  save_video_caps(result, './results/' + name + '_')
  save_video_caps(A, './results/' + name + '_seams_')
  cap.release()
  print 'Finished file: ' + basename(filename)


methods_batch = ([], [])
for filename in glob.glob("./" + args.path + "*"):
  methods_batch[0].append(batch_videos)
  methods_batch[1].append((filename,))

print methods_batch[0], methods_batch[1]
parallelize(methods_batch[0], methods_batch[1])
