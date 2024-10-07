
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from utils import *


#model = torch.load("10S_nadir.pt", weights_only=False)["model"]
#model = YOLO("10S_nadir.pt")
#model = YOLO("10S_nadir.torchscript", task="detect")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("10S_nadir.torchscript", map_location=device).to(device)
for param in model.parameters():
    print(f"Model parameter is on device: {param.device}")
    break
#model.eval()


img_np = cv2.imread("l8_10S_00004.jpg")
landmark_list = []
#  (batch_size, channels, height, width)
print(f"Original image size: {img_np.shape}")
img = cv2.resize(img_np, (1216,1216))

img = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).float()
print(f"Image size after resizing: {img.shape}")


#img = img.repeat(2, 1, 1, 1)  # Create a batch of 2 by repeating the image
#print(f"Image size after resizing and batching: {img.shape}")



with torch.inference_mode():
    img = img.to(device)
    results = model(img).to(device)

print(results)
print("results shape: ", results.size())

# [batch_size, num_classes+4, num_detections]
# assume batch of one for now 
batch_size = results.shape[0]
N_detections = results.shape[2]
num_classes = results.shape[1] - 4
print(f"Batch size: {batch_size}")
print(f"Number of classes: {num_classes}")
print(f"Number of detections: {N_detections}")

## get the array of confidences and plot?
confidences = results[0, 4, :]

# Save the confidences in a csv file 
np.savetxt("confidences.csv", confidences.cpu().numpy(), delimiter=",")
import nms_new
preds = nms_new.non_max_suppression(
    results,
    conf_thres=0.25,
    iou_thres=0.45,
)
print(preds[0])
print(preds[0].shape)

orig_img = img_np

results = []
for i, pred in enumerate(preds):
    print(pred.shape)
    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)


    results.append(pred)

print(results)
print(results[0].shape)
results[0] = results[0].to("cpu")
# detections (4) + conf (1) + class (1) 
#exit()
### 
# Process each detection result from the model
for result in results:
    landmarks = result

    # Iterate over each detected bounding box (landmark)
    for landmark in landmarks:
        x, y, w, h = landmark[:4]
        cls = landmark[5]
        conf = landmark[4]
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


"""landmarks = []

class Landmark:
    def __init__(self, xywh, cls, conf):
        self.xywh = xywh
        self.cls = cls
        self.conf = conf


# Prepare data for NMS
bboxes = []
scores = []

for i in range(N_detections):
    bbox = results[0, :4, i]  # Extract the bounding box coordinates (x, y, w, h)
    class_probs = results[0, 4:, i]  # Extract class probabilities
    class_id = torch.argmax(class_probs).item()  # Get the class with the highest probability
    conf = class_probs[class_id].item()  # Get the corresponding confidence score

    if conf > 0.5:  # Confidence threshold
        # Convert bbox to format required by NMS (x, y, w, h -> x1, y1, x2, y2)
        x, y, w, h = bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        bboxes.append([x1, y1, x2, y2])
        scores.append(conf)
        landmarks.append(Landmark(bbox, class_id, conf))

# Convert to format for OpenCV NMS
bboxes = np.array(bboxes)
scores = np.array(scores)



# Apply NMS
nms_threshold = 0.45  # Intersection-over-union threshold for NMS
score_threshold = 0.25  # Filter out boxes with confidence score less than this
indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), score_threshold, nms_threshold)

# Process kept boxes
for i in indices:
    landmark = landmarks[i]
    print(f"Kept box: {landmark.xywh}, class: {landmark.cls}, confidence: {landmark.conf}")


print(f"Kept {len(indices)} boxes after NMS")


landmarks = [landmarks[i] for i in indices.flatten()]



# Process each detection result from the model
#for result in results:
for r in range(batch_size):
    #landmarks = result.boxes

    # Iterate over each detected bounding box (landmark)
    for landmark in landmarks:

        x, y, w, h = landmark.xywh
        cls = landmark.cls
        conf = landmark.conf
        

        # Validate bounding box dimensions (e.g., non-negative)
        if w < 0 or h < 0:
            print("Invalid bounding box dimensions detected.")
            continue

        # Validate confidence level (e.g., consider only high confidence detections)
        if conf < 0.5:  
            print("Skipping low confidence landmark.")
            continue

        landmark_list.append(
            [
                int(x),
                int(y),
                cls,
                int(w),
                int(h),
            ]
        )


landmark_arr = np.array(landmark_list)
print(f"landmark array shape: {landmark_arr.shape}")

if landmark_arr.shape[0] == 0:
    print("No landmarks detected.")
    exit()"""


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
    cv2.rectangle(img_np, top_left, bottom_right, (0, 255, 0), 2)
    
    # Draw the centroid
    cv2.circle(img_np, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Put the class label near the centroid
    cv2.putText(img_np, str(cls), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with drawn landmarks
img_np = cv2.resize(img_np, (800, 800))
cv2.imshow("Landmarks", img_np)
cv2.waitKey(0)
cv2.destroyAllWindows()


