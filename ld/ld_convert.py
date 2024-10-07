import torch
from ultralytics import YOLO

def convert_model(path):
    model = YOLO(path) 
    state_dict = model.model.state_dict()
    model.model.load_state_dict(state_dict, strict=False)
    model.model.eval()
    model.export(format="torchscript", imgsz=1216)



if __name__ == "__main__":
    path = "10S_nadir.pt"
    convert_model(path)

    #torchscript_model = YOLO("24T_nadir.torchscript")
    model = YOLO("10S_nadir.torchscript", task="detect")