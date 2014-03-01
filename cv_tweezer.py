import cv2
import cv2.cv
import numpy as np

if __name__ == "__main__":
    background = cv2.imread("background20.bmp", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("blocking7.bmp", cv2.IMREAD_GRAYSCALE)
    result = cv2.addWeighted(image, 1.0, background, -1.0, 0)
    result = cv2.blur(result, (5,5))
    cv2.imwrite("result.bmp", result)
    ret, result2 = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite("result2.bmp", result2)
    circles = cv2.HoughCircles(result2, cv2.cv.CV_HOUGH_GRADIENT, 1, 1, param1=50, param2=20, minRadius = 10, maxRadius = 60)
    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2RGB)
    circles = np.uint16(np.around(circles))
    #circles = circles[0, :] 
    circles = [max(circles[0, :].tolist(), key = lambda x: x[2])]
    for i in circles:
        cv2.circle(result2, (i[0], i[1]), i[2], (0,255,0), 2)
        cv2.circle(result2, (i[0], i[1]), 2, (0,255,255), 3)
    #result2 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                cv2.THRESH_BINARY, 31, -35)
    cv2.imwrite("result3.bmp", result2)
