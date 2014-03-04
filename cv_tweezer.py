import cv2
import cv2.cv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
class Frame:
    def __init__(self, path, protocol):
        self.path = path
        self.protocol = protocol
        self.image = None
        self.center = None
        self.radius = None
    def read(self):
        self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
    def obtain_center(self):
        self.protocol(self)
    def save(self, new_path = 'default'):
        if new_path == 'default':
            cv2.imwrite(self.path, self.image)
        else:
            cv2.imwrite(new_path, self.image)
    def free(self):
        del self.image
        self.image = None

class Animation:
    def __init__(self):
        self.frames = []
        self.x = []
        self.y = []
        self.x_sd = None
        self.y_sd = None
    def import_frame(self, path, protocol):
        self.frames.append(Frame(path, protocol))
    def import_seq(self, path, regex, protocol):
        self.frames = []
        indice = []
        for file_name in os.listdir(path):
            reg_match = re.match(regex, file_name)
            if reg_match != None:
                self.frames.append(Frame(path+file_name, protocol))
                indice.append(int(reg_match.group(1)))
        self.frames = [frame for (index, frame) in sorted(zip(indice, self.frames))]
    def append(self,frame):
        self.frames.append(frame)
    def analyze(self, norm_factor = 1.0):
        centers = []
        for frame in self.frames:
            frame.obtain_center()
            centers.append(map(lambda x:x*norm_factor, frame.center))
        xs, ys = zip(*centers)
        self.x_sd = np.std(xs)
        self.y_sd = np.std(ys)
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        self.x = map(lambda x:x-mean_x, xs)
        self.y = map(lambda y:y-mean_y, ys)
