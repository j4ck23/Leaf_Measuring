import cv2
import numpy as np

# Load image
image = cv2.imread("Row_9_Whole_Bag_11-06-26.jpg")

# Resize
scale = 0.5
image = cv2.resize(image, None, fx=scale, fy=scale)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Green threshold
lower_green = np.array([40, 50, 40])
upper_green = np.array([85, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Count pixels
green_pixels = cv2.countNonZero(mask)
total_pixels = mask.shape[0] * mask.shape[1]
non_green_pixels = total_pixels - green_pixels

green_percent = (green_pixels / total_pixels) * 100
non_green_percent = (non_green_pixels / total_pixels) * 100

#Output results displaying the number of green pixels, non-green pixels, and their respective percentages
print(f"Green Pixels: {green_pixels:,}")
print(f"Non-Green Pixels: {non_green_pixels:,}")
print(f"Green Coverage: {green_percent:.2f}%")
print(f"Non-Green Coverage: {non_green_percent:.2f}%")

green_only = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Green Areas", green_only)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = image.copy()
result[mask > 0] = [0, 255, 0]  # BGR = Green
cv2.imshow("Green Mask", result)
cv2.waitKey(0)
cv2.destroyAllWindows()