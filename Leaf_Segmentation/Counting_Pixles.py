import cv2
import numpy as np
from ultralytics import YOLO
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

# -------------------------
# Load segmentation model
# -------------------------
model = YOLO("runs/segment/train/weights/best.pt")   # Your trained segmentation model

# -------------------------
# Load image
# -------------------------
image = cv2.imread("Row_9_Whole_Bag_11-06-26.jpg")
#image = cv2.imread("Row_9_P_1_Front_11-06-26.jpg")

scale = 0.5
image = cv2.resize(image, None, fx=scale, fy=scale)

# Run inference
results = model(image)

result = results[0]
annotated_image = results[0].plot(
    boxes=False,
)

leaf_count = 0
strawberry_count = 0
flowers_count = 0
for cls, mask, masks in zip(result.boxes.cls.cpu().numpy(), results[0].masks.data, results[0].masks):
    if int(cls) == 1:
        leaf_count += 1 
        num_pixels = mask.sum()
        polygon = masks.xy[0].astype(np.int32)
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon[0]
        print(f"Leaf {leaf_count} pixel count: {num_pixels}")
        cv2.putText(annotated_image, f"L:{leaf_count}: {num_pixels}px", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    elif int(cls) == 2:
        strawberry_count += 1
        num_pixels = mask.sum()
        polygon = masks.xy[0].astype(np.int32)
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon[0]
        print(f"Strawberry {strawberry_count} pixel count: {num_pixels}")
        cv2.putText(annotated_image, f"S:{strawberry_count}: {num_pixels}px", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    elif int(cls) == 3:
        flowers_count += 1
        num_pixels = mask.sum()
        polygon = masks.xy[0].astype(np.int32)
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon[0]
        print(f"Flower {flowers_count} pixel count: {num_pixels}")
        cv2.putText(annotated_image, f"F:{flowers_count}: {num_pixels}px", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)




cv2.putText(annotated_image, f"Leaves: {leaf_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(annotated_image, f"Strawberries: {strawberry_count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(annotated_image, f"Flowers: {flowers_count}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0) 

tracker.stop()