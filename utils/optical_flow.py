import cv2
import numpy as np

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

def farneback(previous_frame, frame):
  flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, 0.5, 1, 3, 15, 3, 5, 1)
  return generate_flow_map(flow)
