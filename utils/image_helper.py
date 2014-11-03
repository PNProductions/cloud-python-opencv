import cv2
import os.path
from numpy import clip


def local_path(path):
  return os.path.dirname(__file__) + '/' + path


def image_open(filename, mode=None):
  if os.path.isfile(filename):
    if mode is None:
      image = cv2.imread(filename)
    else:
      image = cv2.imread(filename, mode)
    if image is None:
      IOError('Unable to open image file: ' + filename)
    else:
      return image
  else:
    raise IOError('Image file not found at: ' + filename)


def image_save(image, name, path='./', extension='.bmp'):
  cv2.imwrite(path + name + extension, image)


def to_matlab_ycbcr(image):
  # http://stackoverflow.com/questions/26078281/why-luma-parameter-differs-in-opencv-and-matlab
  return clip(16 + (219 / 255.0) * image, 0, 255)


def from_matlab_ycbcr(image):
  # http://stackoverflow.com/questions/26078281/why-luma-parameter-differs-in-opencv-and-matlab
  # return clip(image * (255 / 219.0) - 16, 0, 255)
  return clip(image * (255 / 219.0) - 16, 0, 255)
