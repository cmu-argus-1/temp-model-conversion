from ultralytics import YOLO
import cv2

path = "10S_nadir.pt"

model = YOLO(path)
model.export(format="engine", imgsz=1216,  half=True, device="cuda:0", workspace=4, nms=True)



img_np = cv2.imread("l8_10S_00004.jpg")
landmark_list = []
#  (batch_size, channels, height, width)
print(f"Original image size: {img_np.shape}")
img = cv2.resize(img_np, (1216,1216))


tensorrt_model = YOLO(path + ".engine")
results = tensorrt_model(img)




