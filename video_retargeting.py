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
import seammerging as sm
from utils.optical_flow import farneback
from utils.video_helper import save_video_caps
from utils.parallelize import parallelize
from utils.image_helper import image_open, image_save
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

parser = argparse.ArgumentParser(description='Image and video retargeting with different methods')
parser.add_argument('-p', '--path', default='testing_videos/downsample/japan/', type=str, help='Path for the videos to retarget')
parser.add_argument('-s', '--seam', type=int, help='Seam to use (default 1 for videos, image_width/2 for images)')
parser.add_argument('-f', '--frame', type=int, help='Frame to use (default all video frame)')
parser.add_argument('-i', '--save-importance', action='store_true', help='Save importance map')
parser.add_argument('-nv', '--no-save-vectors', default=True, action='store_false', dest='save_vectors', help='Do not save motion vector')
parser.add_argument('-g', '--global-vector', action='store_true', help='Use global motion vector')
parser.add_argument('-m', '--method', default='seam_merging_gc', choices=['seam_merging_gc', 'seam_merging', 'seam_carving', 'time_merging'], help='Algorithm to use, seam_merging_gc is the seam merging algorithm using graph cut instead of dynamic programming')
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

def get_cap(filename):
  if filename.endswith(('.m4v')):
    return cv2.VideoCapture(filename)
  return image_open(filename)

def get_output_file_name(filename):
  size = '_reduce' if args.seam < 0 else '_enlarge'
  size += str(-args.seam) if args.seam < 0 else str(args.seam)
  return splitext(basename(filename))[0] + suffix + '_' + '_' + size + '_' + args.method

def cartoon_image(image):
  y = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB).astype(np.float64)
  return y, TotalVariationDenoising(y[:, :, 0], iterTV).generate()

def importance_map(image):
  kernel = np.array([[0, 0, 0],
                        [1, 0, -1],
                        [0, 0, 0]
                        ])
  return np.abs(cv2.filter2D(image[:, :, 0], -1, kernel, borderType=cv2.BORDER_REPLICATE)) + np.abs(cv2.filter2D(image[:, :, 0], -1, kernel.T, borderType=cv2.BORDER_REPLICATE))

def batch(filename):
  if filename.endswith(('.m4v')):
    batch_videos(filename)
  else:
    batch_images(filename)

def batch_images(filename):
  image = get_cap(filename)
  args.seam = args.seam if args.seam != None else -image.shape[1]/2
  name = get_output_file_name(filename)
  y, cartoon = cartoon_image(image)
  importance = importance_map(y)
  if args.method == "seam_merging_gc":
    img = vs.seam_merging(image, cartoon, importance, None, args.seam if args.seam != None else image.shape[0]/2, alpha, beta)
  elif args.method == "seam_merging":
    img = sm.seam_merging(image, cartoon, importance, args.seam, alpha, beta)
  elif args.method == "seam_carving":
    img = sc.seam_carving(image, args.seam)
  else:
    sys.exit("Method " + args.method + " couldn't be applied to " + filename)
  image_save(img, name, './results/')
  print filename + ' finished!'

def batch_videos(filename):
  cap = get_cap(filename)
  args.seam = args.seam if args.seam != None else 1
  name = get_output_file_name(filename)

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
      ret, image = cap.read()
      if not ret:
        break
      y, cartoon[i] = cartoon_image(image)
    
      # Motion vector frame per frame (filtered with medianBlur to delete salt and pepper noise)
      if i > 0:
        vectors[i - 1] = farneback(cartoon[i - 1], cartoon[i])
        vectors[i - 1] = cv2.medianBlur(vectors[i - 1].astype(np.uint8), 19).astype(np.float64)

      importance[i] = importance_map(y)

      video[i] = image
      i += 1

    savemat(mat_name, {'importance': importance, 'cartoon': cartoon, 'video': video, 'vectors':vectors}, do_compression=True)

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

  if args.method == "seam_merging_gc":
    result, seams = vs.seam_merging(video, cartoon, importance, vectors, args.seam, alpha, beta)
  elif args.method == "seam_carving":
    result, seams = sc.seam_carving(video, args.seam)
  elif args.method == "time_merging":
    result, seams = vs.time_merging(video, cartoon, importance, None, args.seam, alpha, beta)
  else:
    sys.exit("Method " + args.method + " couldn't be applied to " + filename)

  # Saving seams frame images
  A = print_seams(video, seams)
  save_video_caps(np.clip(result, 0, 255).astype(np.uint8), './results/' + name + '_')
  save_video_caps(np.clip(A * 0.8 + video, 0, 255).astype(np.uint8), './results/' + name + '_seams_')
  cap.release()
  print 'Finished file: ' + basename(filename)

batch("./testing_videos/downsample/other/car_down_61.m4v")

# methods_batch = ([], [])
# for filename in glob.glob("./" + args.path + "/*"):
#   methods_batch[0].append(batch)
#   methods_batch[1].append((filename,))

# parallelize(methods_batch[0], methods_batch[1])
