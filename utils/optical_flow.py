import cv2
import numpy as np


def norm2(x, y):
  return np.sqrt(x ** 2 + y ** 2)


def generate_flow_map(flow):
  h, w = flow.shape[0], flow.shape[1]
  # Get a list of indexes starting from step/2 to the end of the image, with step = step (16)
  y, x = np.mgrid[0:h, 0:w]
  fy, fx = flow[y, x].T
  fy, fx = fy.T, fx.T

  min_square = np.sqrt(fx ** 2 + fy ** 2)
  result = np.zeros((flow.shape[0], flow.shape[1]))
  dy, dx = np.clip(y + fy, 0, h - 1), np.clip(x + fx, 0, w - 1)

  to_take = np.where(min_square >= 1)
  y, x, dy, dx = y[to_take], x[to_take], dy[to_take], dx[to_take]

  result[np.int32(y), np.int32(x)] += min_square[np.int32(y), np.int32(x)]
  result[np.int32(dy), np.int32(dx)] += min_square[np.int32(y), np.int32(x)]
  return result


def estimate_global_motion(previous_frame, frame):
  affine_transform = cv2.estimateRigidTransform(previous_frame.astype(np.uint8), frame.astype(np.uint8), False)
  affine_transform = np.zeros((2, 3)) if affine_transform is None else affine_transform
  return affine_transform[:, -1].T


def apply_translation(previous_frame, frame, global_motion):
  dx, dy = global_motion.astype(np.int32)

  f1, f2 = np.zeros_like(previous_frame), np.zeros_like(frame)
  if dx > 0:
    # Se in x da 2 va in 5, devo tirarlo in dietro. In f2[0] metto frame[3]
    # Per tenere coerenza, metto a 0 le linee di destra
    f1[:, :-dx] = previous_frame[:, :-dx]
    f2[:, :-dx] = frame[:, dx:]
  elif dx < 0:
    # Se X da 5 va in 2 devo portarlo avanti di 3. In f2[3] metto frame[0]
    # Per tenere coerenza, metto a 0 le linee di sinistra
    f1[:, -dx:] = previous_frame[:, -dx:]
    f2[:, -dx:] = frame[:, :dx]
  else:
    f1, f2 = previous_frame, frame

  previous_frame = f1
  frame = f2

  f1, f2 = np.zeros_like(previous_frame), np.zeros_like(frame)
  if dy > 0:
    f1[:-dy] = previous_frame[:-dy]
    f2[:-dy] = frame[dy:]
  elif dy < 0:
    f1[-dy:] = previous_frame[-dy:]
    f2[-dy:] = frame[:dy]
  else:
    f1, f2 = previous_frame, frame

  return f1, f2


def farneback(previous_frame, frame):
  flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, 0.5, 1, 3, 15, 3, 5, 1)
  return generate_flow_map(flow)


def improved_farneback(previous_frame, frame):
  global_motion = estimate_global_motion(previous_frame, frame)
  camera_moved = norm2(*global_motion) > 2
  if camera_moved:
    print 'corrected', global_motion
    previous_frame, frame = apply_translation(previous_frame, frame, global_motion)
    flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, 0.5, 1, 3, 15, 3, 5, 1)
  else:
    flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, 0.5, 1, 3, 15, 3, 5, 1)
  return generate_flow_map(flow)
