import numpy as np


def generate_step(I, img):
  result = np.empty((img.shape[0], img.shape[1], img.shape[2]))
  r = np.arange(img.shape[2])
  for i in xrange(img.shape[0]):
    result[i] = r > np.vstack(I[i])
  return result.astype(np.uint64)


def print_seams3(result, seams, amount):
  seams = seams.astype(np.uint64)
  A = np.zeros_like(result)
  correction = np.zeros(result.shape[:3], dtype=np.uint64)
  iterator = reversed(xrange(seams.shape[0])) if amount > 0 else xrange(seams.shape[0])
  for i in iterator:
    X, Y = np.mgrid[:result.shape[0], :result.shape[1]]
    I = seams[i]
    I = I + correction[X, Y, I]
    color = np.random.rand(3) * 255
    A[X, Y, I] = color
    correction = correction + generate_step(I, result)
  return A * 0.8 + result


def print_seams2(result, seams, amount):
  seams = seams.astype(np.uint64)
  A = np.zeros_like(result)
  correction = np.zeros(result.shape[:2], dtype=np.uint64)
  iterator = reversed(xrange(seams.shape[0])) if amount > 0 else xrange(seams.shape[0])
  for i in iterator:
    Y = np.r_[:result.shape[0]]
    I = seams[i]
    I = I + correction[Y, I]
    color = np.random.rand(3) * 255
    A[Y, I] = color
    c = np.r_[:result.shape[1]] > np.vstack(I)
    correction = correction + c.astype(np.uint64)
  return A * 0.8 + result


def print_seams(original, result, seams, amount):
  target = original if amount < 0 else result
  return print_seams3(target, seams, amount) if seams.ndim == 3 else print_seams2(target, seams, amount)


def generate_time_step(I, img):
  result = np.empty(img.shape[:3])
  r = np.arange(img.shape[0])
  for i in xrange(img.shape[1]):
    result[:, i] = (r > np.vstack(I[i])).T
  return result.astype(np.uint64)


def print_time_seams(result, seams):
  seams = seams.astype(np.uint64)
  A = np.zeros_like(result)
  correction = np.zeros(result.shape[:3], dtype=np.uint64)
  for i in xrange(seams.shape[0]):
    X, Y = np.mgrid[:result.shape[1], :result.shape[2]]
    I = seams[i]
    I = I + correction[I, X, Y]
    color = np.random.rand(3) * 255
    A[I, X, Y] = color
    correction = correction + generate_time_step(I, result)
  return A
