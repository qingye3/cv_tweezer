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
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
        for file_name in os.listdir(path):
            if re.match(regex, file_name) != None:
                self.frames.append(Frame(path+file_name, protocol))
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

#def scatter_plot(animations, path):
#    plt.figure()
#    for i,animation in enumerate(animations):
#        x, y = zip(*animation.centers)
#
#        plt.scatter(        
    

if __name__ == "__main__":
    kernal = np.ones((3,3), np.uint8)
    background = cv2.imread("background20.bmp", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("blocking7.bmp", cv2.IMREAD_GRAYSCALE)
    result = cv2.addWeighted(image, 1.0, background, -1.0, 0)
    result = cv2.blur(result, (5,5))
    cv2.imwrite("result.bmp", result)
    #ret, result2 = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)
    result2 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
    cv2.imwrite("result2.bmp", result2)
    circles = cv2.HoughCircles(result2, cv2.cv.CV_HOUGH_GRADIENT, 1, 5, param1=50, param2=20, minRadius = 10, maxRadius = 60)
    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2RGB)
    circles = np.uint16(np.around(circles))
    #circles = circles[0, :] 
    circles = [min(circles[0, :].tolist(), key = lambda x: x[2])]
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    for i in circles:
        cv2.circle(result, (i[0], i[1]), i[2], (0,255,0), 2)
        cv2.circle(result, (i[0], i[1]), 2, (0,255,255), 3)
        cv2.circle(result2, (i[0], i[1]), i[2], (0,255,0), 2)
        cv2.circle(result2, (i[0], i[1]), 2, (0,255,255), 3)
    #result2 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                cv2.THRESH_BINARY, 31, -35)
    cv2.imwrite("result3.bmp", result)
    cv2.imwrite("result4.bmp", result2)
