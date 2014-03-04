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
        self.blur_kernal    = SD_Protocol.init_mVar(blur_kernal, (5,5))
        self.saved_results  = SD_Protocol.init_mVar(saved_results, [4])
        self.circle_chooser = SD_Protocol.init_mVar(circle_chooser, SD_Protocol.choose_biggest_circle) 
        self.Hough_min_dist = SD_Protocol.init_mVar(Hough_min_dist, 1)
        self.Hough_param1   = SD_Protocol.init_mVar(Hough_param1, 50)
        self.Hough_param2   = SD_Protocol.init_mVar(Hough_param2, 20)
        self.minRad         = SD_Protocol.init_mVar(minRad, 10)
        self.maxRad         = SD_Protocol.init_mVar(maxRad, 60)
        self.thresh_hold    = SD_Protocol.init_mVar(thresh_hold, 30)
        self.adapt_rad      = SD_Protocol.init_mVar(adapt_rad, 51)
        self.adapt_plus     = SD_Protocol.init_mVar(adapt_plus, -10)
        self.thresh_hold_func = SD_Protocol.init_mVar(thresh_hold_func, self.adaptive_thresh_hold)
        self.path           = None
        
    
    def __call__(self, frame):
        self.path = frame.path
        print "Processing %s" %self.path
        frame.read()
        temp_image = np.copy(frame.image)

        temp_image = cv2.addWeighted(temp_image, 1.0, self.background_image, -1.0, 0)
        subtracted_image = np.copy(temp_image)
        self.save(temp_image, "subtracted", 0)

        for i in range(3):
            temp_image = cv2.blur(temp_image, self.blur_kernal )
        self.save(temp_image, "blurred", 1)

        temp_image = self.thresh_hold_func(temp_image)
        self.save(temp_image, "threshod", 2)

        circles = None
        grace = 0
        while circles == None:
            circles = cv2.HoughCircles(temp_image, cv2.cv.CV_HOUGH_GRADIENT, 1, self.Hough_min_dist, param1 = self.Hough_param1, param2 = self.Hough_param2 - grace, minRadius = self.minRad, maxRadius = self.maxRad)
            grace +=1
        circles = np.uint16(np.around(circles))
        x, y, r = self.circle_chooser(circles)

        temp = np.copy(temp_image)
        contours, garbage = cv2.findContours(temp, cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_NONE)
        circles2 = map(cv2.minEnclosingCircle, contours)
        (x, y), r = min(circles2, key = lambda x: abs(r-x[1]))
        frame.center = (x,y)
        frame.radius = r

        x, y, r = map(lambda m: int(round(m)), [x,y,r])

        subtracted_image= SD_Protocol.toRGB(subtracted_image)
        SD_Protocol.draw_circle(subtracted_image, x, y, r)
        self.save(subtracted_image, "circle", 3)

        frame.free()
        del temp_image
        temp_image = None

    def adaptive_thresh_hold(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.adapt_rad, self.adapt_plus)

    def global_thresh_hold(self, image):
        return cv2.threshold(image, self.thresh_hold, 255, cv2.THRESH_BINARY)

        
    def save(self,image, note, i):
        if i in self.saved_results:
            cv2.imwrite(get_save_path(self.path, note), image)

    @staticmethod 
    def draw_circle(image, x, y, r):
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 255, 0), 3)

    @staticmethod
    def toRGB(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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

    @staticmethod
    def choose_first_circle(circles):
        return circles[0, :].tolist()[0]

def plotsd(experiments):
    size = len(experiments)
    plt.figure(1)
    plt.title("SD vs Blocking Level")
    plt.xlabel("Blocking Level")
    plt.ylabel("SD")
    plt.plot(range(size), [exp.x_sd for exp in experiments], '-b', label='sd of x')
    plt.plot(range(size), [exp.y_sd for exp in experiments], '-r', label='sd of y')
    plt.savefig("../results/sd_vs_blocking.png")

def plotscatter(experiments, dist = 15):
    size = len(experiments)
    colors = ['r', 'b', 'g', 'k', 'y', 'c', 'm', 'r' ]
    plt.figure(2)
    plt.title("Scatter")
    for i, exp in enumerate(experiments):
        x = [e_x + i * dist for e_x in exp.x]
        y = [e_y + i * dist for e_y in exp.y]
        plt.scatter(x, y, color = colors[i], s=2)
    plt.savefig("../results/scatter.png")
    

    
if __name__ == "__main__":
    background = cv2.imread("background20.bmp", cv2.IMREAD_GRAYSCALE)
    protocol = SD_Protocol(background, saved_results = range(4))
    experiments = []
    for i in range(8):
        experiment = Animation()
        experiment.import_seq("../Blocking%d/" % i, "blocking(\d+).bmp\Z", protocol)
        experiment.analyze()
        experiments.append(experiment)
    plotscatter(experiments)
    plotsd(experiments)
    


