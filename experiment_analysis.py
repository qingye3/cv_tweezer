import cv2
import cv2.cv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from cv_tweezer import Frame, Animation

def get_save_path(path, note = "result"):
    abspath = os.path.abspath(path)
    splitpath = abspath.split('/')
    try:
        os.makedirs('../results/'+splitpath[-2])
    except OSError:
        pass
    retpath = "/".join(splitpath[-2:])
    retpath = retpath[:-4] + "_%s" %note + ".bmp"
    retpath = "../results/" + retpath
    return retpath

class SD_Protocol:
    def __init__(self, background_image_arr, blur_kernal = None, saved_results = None, circle_chooser = None, Hough_param1 = None, Hough_param2 = None, Hough_min_dist = None, minRad = None, maxRad = None, thresh_hold_func = None, thresh_hold = None, adapt_rad = None, adapt_plus = None ):
        self.background_image = background_image_arr
        self.blur_kernal    = init_mVar(blur_kernal, np.ones((5,5), np.uint8))
        self.saved_results  = init_mVar(saved_results, [])
        self.circle_chooser = init_mVar(circle_chooser, choose_smallest_circle) 
        self.Hough_min_dist = init_mVar(Hough_min_dist, 1)
        self.Hough_param1   = init_mVar(Hough_param1, 50)
        self.Hough_param2   = init_mVar(Hough_param2, 20)
        self.minRad         = init_mVar(minRad, 10)
        self.maxRad         = init_mVar(maxRad, 60)
        self.thresh_hold    = init_mVar(thresh_hold, 30)
        self.adapt_rad      = init_mVar(adapt_rad, 51)
        self.adapt_plus     = init_mVar(adapt_plus, -10)
        self.thresh_hold_func = init_mVar(thresh_hold_func, self.gaussian_thresh_hold)
        self.path           = None
        
    
    def __call__(self, frame):
        self.path = frame.path
        frame.read()
        temp_image = np.copy(frame.image)

        temp_image = cv2.addWeighted(temp_image, 1.0, self.background_image, -1.0, 0)
        self.save(temp_image, "subtracted", 0)

        temp_image = cv2.blur(temp_image, self.blur_kernal)
        self.save(temp_image, "blurred", 1)

        temp_image = self.thresh_hold_func(temp_image)
        self.save(temp_image, "threshod", 2)

        circles = cv2.HoughCircles(temp_image, cv2.cv.CV_HOUGH_GRADIENT, 1, self.Hough_min_dist, param1 = self.Hough_param1, param2 = self.Hough_param2, minRadius self.minRad, maxRadius = self.maxRad)
        x, y, r = self.circle_chooser(circles)

        toRGB(temp_image)
        draw_circle(temp_image, x, y, r)
        self.save(temp_image, "circle", 3)

        toRGB(frame.image)
        draw_circle(frame.image, x, y, r)
        self.save(frame.image, "circle_on_orig", 4)

        frame.center = (x,y)
        frame.radius = r

        
    def save(image, note, i):
        if i in self.saved_results:
            cv2.imwrite(get_save_path(self.path, note), image)

    @staticmethod 
    def draw_circle(image, x, y, r):
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 255, 0), 3)

    @staticmethod
    def toRGB(image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def init_mVar(value, default):
        if value == None:
            return default
        else:
            return value

    @staticmethod
    def choose_smallest_circle(circles):
        return min(circles[0, :].tolist(), key = lambda x:x[2])

    @staticmethod
    def choose_biggest_circle(circles):
        return max(circles[0, :].tolist(), key = lambda x:x[2])
    
if __name__ == "__main__":
    protocol0 = SD_Protocol()
    exp_0 = Animation()
    exp_0.import_frame("../Blocking0/blocking0.bmp", protocol0)
