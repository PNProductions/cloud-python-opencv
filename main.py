#!/usr/bin/env python

# to kill this process use pkill -f main.py

from os.path import basename, splitext, isfile
import glob
import os
import datetime
import argparse
import sys
import distutils.core
import time
import numpy as np
import cv2
from scipy.io import savemat, loadmat
import videoseam as vs
import seamcarving as sc
import seammerging as sm
from utils.optical_flow import farneback, improved_farneback
from utils.video_helper import save_video_caps
from utils.parallelize import parallelize
from utils.image_helper import image_open, image_save
from utils.seams import print_time_seams, print_seams
from tvd import TotalVariationDenoising

# Fix path name relationships when this script is running from a different folder
if __name__ == '__main__':
  if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))


vs.PROGRESS_BAR = False
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
parser.add_argument('-fs', '--frame-start', type=int, default=1, help='Starting frame')
parser.add_argument('-fe', '--frame-end', type=int, help='Ending frame (default last video frame)')
parser.add_argument('-i', '--save-importance', action='store_true', help='Save importance map')
parser.add_argument('-nv', '--no-save-vectors', default=True, action='store_false', dest='save_vectors', help='Do not save motion vector')
parser.add_argument('-d', '--dropbox', default=False, action='store_true', dest='with_dropbox', help='Remove result folder and copy results in the dropbox folder (only cloud server)')
parser.add_argument('-if', '--improved-farneback', default=False, action='store_true', help='Use improved farneback to check camera movement in the image')
parser.add_argument('-g', '--global-vector', action='store_true', help='Use global motion vector instead that frame per frame vector')
parser.add_argument('-gl', '--global-local-vector', action='store_true', help='Use the sum of global motion vector and the frame per frame vector')
parser.add_argument('-mf', '--method-fix', action='store_true', help='Use the slower but more accurate method for graph construction, only for seam_merging_gc')
parser.add_argument('-m', '--method', default='seam_merging_gc', choices=['seam_merging_gc', 'seam_merging', 'seam_carving', 'time_merging'], help='Algorithm to use, seam_merging_gc is the seam merging algorithm using graph cut instead of dynamic programming')
args = parser.parse_args()


# -------------------------------------------------------
# -------------------------------------------------------
def get_cap(filename):
  if filename.endswith(('.m4v')):
    return cv2.VideoCapture(filename)
  return image_open(filename)


def save_frames(mat, filename, string_append):
  mat = np.clip(mat, 0, 255).astype(np.uint8)
  mat = np.expand_dims(mat, axis=3)[:, :, :, [0, 0, 0]]
  save_video_caps(mat, './results/' + filename + string_append)


def get_output_file_name(filename):
  size = '_reduce' if args.seam < 0 else '_enlarge'
  size += str(-args.seam) if args.seam < 0 else str(args.seam)
  if args.global_vector:
    mv = '_global_'
  elif args.global_local_vector:
    mv = '_global_local_'
  else:
    mv = '_local_mv_'

  return splitext(basename(filename))[0] + suffix + '_' + '_' + size + '_' + mv + args.method


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
  args.seam = args.seam if args.seam is not None else -image.shape[1] / 2
  name = get_output_file_name(filename)
  y, cartoon = cartoon_image(image)
  importance = importance_map(y)
  if args.method == "seam_merging_gc":
    if args.method_fix:
      img, seams = vs.seam_merging(image, cartoon, importance, None, args.seam, alpha, beta, methods=vs.STR | vs.ITE | vs.IMP | vs.FIX)
    else:
      img, seams = vs.seam_merging(image, cartoon, importance, None, args.seam, alpha, beta)
  elif args.method == "seam_merging":
    img, seams = sm.seam_merging(image, cartoon, importance, args.seam, alpha, beta)
  elif args.method == "seam_carving":
    img, seams = sc.seam_carving(image, args.seam, False)
  else:
    sys.exit("Method " + args.method + " couldn't be applied to " + filename)
  image_save(img, name, './results/')
  seams = print_seams(image, img, seams, args.seam)
  image_save(seams, name + '_seams_', './results/')
  print filename + ' finished!'


def batch_videos(filename):
  def normalize_max(mat):
    maxv = mat.max(axis=0)
    maxv[np.where(maxv == 0)] = 1
    return mat[:] / maxv * 255

  cap = get_cap(filename)
  args.seam = args.seam if args.seam is not None else 1
  name = get_output_file_name(filename)

  frame_count, fps, width, height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), cap.get(cv2.cv.CV_CAP_PROP_FPS), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
  frame_count = frame_count if args.frame_end is None else args.frame_end

  importance = np.empty((frame_count - args.frame_start + 1, height, width))
  cartoon = np.empty((frame_count - args.frame_start + 1, height, width))
  video = np.empty((frame_count - args.frame_start + 1, height, width, 3))
  vectors = np.empty((frame_count - args.frame_start + 1, height, width))

  mat_name = './temp/' + splitext(basename(filename))[0] + '_' + str(frame_count) + '_garbage.mat'

  if isfile(mat_name):
    r = loadmat(mat_name)
    video, cartoon, importance, vectors = r['video'], r['cartoon'], r['importance'], r['vectors']
  else:
    i, j = 0, 0
    while cap.isOpened() and j < frame_count:
      j += 1
      ret, image = cap.read()
      if j >= args.frame_start:
        if not ret:
          break

        y, cartoon[i] = cartoon_image(image)

        # Motion vector frame per frame (filtered with medianBlur to delete salt and pepper noise)
        if i > 0:
          if args.improved_farneback:
            vectors[i - 1] = improved_farneback(cartoon[i - 1], cartoon[i])
          else:
            vectors[i - 1] = farneback(cartoon[i - 1], cartoon[i])
          vectors[i - 1] = cv2.medianBlur(vectors[i - 1].astype(np.uint8), 19).astype(np.float64)

        importance[i] = importance_map(y)

        video[i] = image
        i += 1

    savemat(mat_name, {'importance': importance, 'cartoon': cartoon, 'video': video, 'vectors': vectors}, do_compression=True)

  # Motion vector normalizing and saving
  vectors = normalize_max(vectors)
  # Use a global motion vector (consider every frame)
  if args.global_vector or args.global_local_vector:
    normalized = vectors.sum(axis=0)
    normalized = np.clip(normalized, 0, 255)
    if args.global_vector:
      vectors[:] = normalized
      if args.save_vectors:
        vecs = normalized[:, :, np.newaxis]
        cv2.imwrite('./results/' + name + '_vectors.bmp', vecs[:, :, [0, 0, 0]])
    elif args.global_local_vector:
      vectors[:] += normalized
      vectors = normalize_max(vectors)

  if args.save_vectors and not args.global_vector:
      save_frames(vectors, name, '_vectors_')

  if args.save_importance:
    save_frames(importance, name, '_importance_')

  if args.method == "seam_merging_gc":
    # result, seams = vs.seam_merging(video, cartoon, importance, vectors, args.seam, alpha, beta, methods=vs.STR | vs.ITE | vs.IMP) # disattiva i motion vector
    result, seams = vs.seam_merging(video, cartoon, importance, vectors, args.seam, alpha, beta)
  elif args.method == "seam_carving":
    result, seams = sc.seam_carving(video, args.seam, False)
  elif args.method == "time_merging":
    result, seams = vs.time_merging(video, cartoon, importance, vectors, args.seam, alpha, beta)
  else:
    sys.exit("Method " + args.method + " couldn't be applied to " + filename)

  # Saving seams frame images
  if args.method == "time_merging":
    A = print_time_seams(video, seams)
  else:
    A = print_seams(video, result, seams, args.seam)

  save_video_caps(np.clip(result, 0, 255).astype(np.uint8), './results/' + name + '_')
  save_video_caps(np.clip(A, 0, 255).astype(np.uint8), './results/' + name + '_seams_')
  cap.release()
  print 'Finished file: ' + basename(filename)


if args.with_dropbox:
  print "Removing old results"
  os.system("rm -rf results/*")


# batch("./testing_videos/downsample/other/4_videos/car_down_31.m4v")
# batch("./testing_images/reduction/cup.png")

print "----------START----------"
time1 = time.time()

if os.path.isfile(args.path):  # if path is a single file
  batch(args.path)
else:  # if path is folder
  methods_batch = ([], [])
  for filename in glob.glob("./" + args.path + "/*"):
    methods_batch[0].append(batch)
    methods_batch[1].append((filename,))
  parallelize(methods_batch[0], methods_batch[1])

time2 = time.time()
print 'Elaboration tooks %f seconds' % (time2 - time1)
print "----------END----------"

if args.with_dropbox:
  datetime_dir = os.path.join("/home/ec2-user/Dropbox/Tesi cloud", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  os.makedirs(datetime_dir)
  distutils.dir_util.copy_tree(os.path.join(os.getcwd(), "results"), datetime_dir)
