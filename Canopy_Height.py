import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Height_Test.jpg')
img = cv.resize(img, (960, 540))    

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply a threshold to the image to
# separate the objects from the background
ret, thresh = cv.threshold(
    gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# Find the contours of the objects in the image
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Loop through the contours and calculate the area of each object
for cnt in contours:
    area = cv.contourArea(cnt)

    # Draw a bounding box around each
    # object and display the area on the image
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(img, str(area), (x, y),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the final image with the bounding boxes
# and areas of the objects overlaid on top
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()