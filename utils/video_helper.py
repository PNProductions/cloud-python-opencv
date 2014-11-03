import os
import time
from image_helper import image_save


def save_video_caps(video, path='./video_frames_'):
  path = path + str(int(time.time()))
  os.makedirs(path)
  for i in range(video.shape[0]):
    image_save(video[i, :, :, :], 'frame_' + str(i), path + '/')
