
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from utils import *

#model = torch.load("10S_nadir.pt", weights_only=False)["model"]
model = YOLO("10S_nadir.pt")


img = cv2.imread("l8_10S_00004.jpg")
landmark_list = []
#  (batch_size, channels, height, width)
print(f"Original image size: {img.shape}")


with torch.no_grad():
    results = model(img)

print(results[0].boxes.data)
print(results[0].boxes.data.size())

#exit()

# Process each detection result from the model
for result in results:
    landmarks = result.boxes

    # Iterate over each detected bounding box (landmark)
    for landmark in landmarks:
        x, y, w, h = landmark.xywh[0]
        cls = landmark.cls[0].item()
        conf = landmark.conf[0].item()
        #print(cls, conf)

        # Validate bounding box dimensions (e.g., non-negative)
        if w < 0 or h < 0:
            print("Invalid bounding box dimensions detected.")
            continue

        # Validate confidence level (e.g., consider only high confidence detections)
        if conf < 0.5:  # Assuming 0.5 as a threshold for confidence
            print("Skipping low confidence landmark.")
            continue

        landmark_list.append(
            [
                int(x.item()),
                int(y.item()),
                cls,
                int(w.item()),
                int(h.item()),
            ]
        )


landmark_arr = np.array(landmark_list)
print(f"landmark array shape: {landmark_arr.shape}")

if landmark_arr.shape[0] == 0:
    print("No landmarks detected.")
    exit()


# Extract centroid coordinates, class IDs, and dimensions (width and height)
centroid_xy = landmark_arr[:, :2]  # Centroid coordinates [x, y]
landmark_class = landmark_arr[:, 2].astype(int)  # Class IDs as integers
landmark_wh = landmark_arr[:, 3:5]  # Width and height [w, h]

# Calculate the top-left and bottom-right coordinates of the bounding boxes
corner_xy = calculate_bounding_boxes(centroid_xy, landmark_wh)


# Draw bounding boxes and centroids on the image
for i, (x, y) in enumerate(centroid_xy):
    w, h = landmark_wh[i]
    cls = landmark_class[i]
    top_left = (int(x - w / 2), int(y - h / 2))
    bottom_right = (int(x + w / 2), int(y + h / 2))
    
    # Draw the bounding box
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    
    # Draw the centroid
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Put the class label near the centroid
    cv2.putText(img, str(cls), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with drawn landmarks
img = cv2.resize(img, (800, 800))
cv2.imshow("Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()