import sys
ROOT1 = r"C:\Users\anton\Desktop\prova\supervision"
ROOT2 = r"C:\Users\anton\Desktop\prova\supervision\supervision"
sys.path.append(ROOT1)
sys.path.append(ROOT2)
import os
import cv2
import time

def recursive_import(path,child):
    path2add = os.path.join(path,child)
    if os.path.isdir(path2add):
        sys.path.append(path2add)
        for newchild in os.listdir(path2add):
            recursive_import(path2add,newchild)

for path in os.listdir(ROOT2):
    recursive_import(ROOT2,path)
if "supervision" in sys.modules.keys():
    print("ok")
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


if __name__ == "__main__":
    
    model = YOLO('yolov8n.pt').cuda()
    results = model.train(data="./data.yaml",epochs=30,imgsz=320,batch = 16, verbose=True, val=False ,workers=0)