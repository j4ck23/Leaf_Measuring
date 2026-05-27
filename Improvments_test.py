import cv2
import numpy as np

img = cv2.imread("Height_Test.jpg")
img = cv2.resize(img, (1000, 600))

# Convert to HSV
# HSV is more robust to lighting changes and allows us to easily isolate the green vegetation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Green color range
lower_green = np.array([25, 40, 40])
upper_green = np.array([90, 255, 255])

# Vegetation mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Morphological cleanup
# We use a large kernel to fill gaps between leaves and remove small noise, which helps in creating a more accurate contour of the canopy.
kernel = np.ones((2,2), np.uint8)

# Fill gaps between leaves
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Remove tiny noise
# This step is crucial to ensure that we only get contours of the canopy and not small specks of noise, which can lead to inaccurate height measurements.
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter tiny contours
filtered = [
    c for c in contours
    if cv2.contourArea(c) > 5000
]

# Merge contours
all_points = np.vstack(filtered)

# Single convex hull
hull = cv2.convexHull(all_points)

# Draw
output = img.copy()

cv2.drawContours(output, [hull], -1, (0,255,0), 3)

cv2.imshow("Mask", mask)
cv2.imshow("Hull", output)

cv2.waitKey(0)
cv2.destroyAllWindows()