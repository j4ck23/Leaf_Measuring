import cv2 as cv
import numpy as np


img = cv.imread('Height_Test.jpg')
img = cv.resize(img, (960, 540))    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#Apply a threshold to the image to separate the objects from the background
blur = cv.blur(gray, (5, 5))
ret, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hull = []

for i in range(len(contours)):
    hull.append(cv.convexHull(contours[i], False))

drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    color_contours = (0, 255, 0)
    color = (255, 0, 0)
    cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    cv.drawContours(drawing, hull, i, color, 1, 8)

cv.imshow('Convex Hull', drawing)
cv.waitKey(0)