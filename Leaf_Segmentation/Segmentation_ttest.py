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
image = cv2.imread("Row_9_P_1_Front_11-06-26.jpg")

scale = 0.5
image = cv2.resize(image, None, fx=scale, fy=scale)

# Run inference
results = model(image)

result = results[0]
annotated_image = results[0].plot(
    boxes=False,
)


overlay = image.copy()

leaf_count = 0

# -------------------------
# Loop through every detection
# -------------------------
if result.masks is not None:

    masks = result.masks.xy          # Polygon coordinates
    classes = result.boxes.cls.cpu().numpy()

    for polygon, cls in zip(masks, classes):

        # Only count leaves
        if int(cls) != 1:
            continue

        leaf_count += 1

        polygon = polygon.astype(np.int32)

        # Fill polygon onto overlay
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))

        # Draw outline
        cv2.polylines(annotated_image,
                      [polygon],
                      True,
                      (255, 255, 255),
                      2)


# -------------------------
# Display total count
# -------------------------
cv2.putText(annotated_image, f"Leaves: {leaf_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

print(f"Total leaves detected: {leaf_count}")
cv2.imwrite("output_with_counts.jpg", annotated_image)
cv2.imshow("Segmentation", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

tracker.stop()