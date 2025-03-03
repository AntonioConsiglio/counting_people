import sys
ROOT1 = r"C:\Users\anton\Desktop\prova\supervision"
ROOT2 = r"C:\Users\anton\Desktop\prova\supervision\supervision"
sys.path.append(ROOT1)
sys.path.append(ROOT2)
import os
import cv2
import time
from utils import LineCounter
import numpy as np

TEMPLATE = cv2.imread("template.png").reshape((320*240,3))
TEMPLATE_IDX = np.all(TEMPLATE == np.array((255,255,255)),axis=1)
# tosee = (TEMPLATE_IDX*255).astype(np.uint8)
# unique = np.unique(tosee)
# cv2.imshow("template",tosee.reshape(240,320))
# cv2.waitKey(0)
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

class Timer():
    def __init__(self):
        self.time_recorded = []
        self.max_time = 0
        self.min_time = 100000000
        self.tmp_start = 0
    
    def start(self):
        self.tmp_start = time.time() 

    def end(self):
        delay = time.time()-self.tmp_start
        self.time_recorded.append(delay)
        if delay > self.max_time:
            self.max_time = delay
        elif delay < self.min_time:
            self.min_time = delay

    def get_statistics(self):
        print("\n")
        print("{:15} : {:.2f} ms".format("AVG TIME REQUIRED",sum(self.time_recorded)*1000/len(self.time_recorded)))
        print("{:15} : {:.2f} ms".format("MIN TIME REQUIRED",self.min_time*1000))
        print("{:15} : {:.2f} ms".format("MAX TIME REQUIRED",self.max_time*1000))

model = YOLO(r'C:\Users\anton\Desktop\prova\runs\detect\train2\weights\best.pt').cuda()           
line_counter = LineCounter([0,110])

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
track_annotator = sv.TraceAnnotator()

VIDEO_ROOT = r"C:\Users\anton\Desktop\VIDEO_ANNOTATION"
DATAROOT = "./newData"

def calculate_cords(cordinate,h,w):
    x1,y1,x2,y2 = cordinate
    xcenter = ((x2-x1)/2 + x1)/w
    ycenter = ((y2-y1)/2 + y1)/h
    hd = (y2-y1)/h
    wd = (x2-x1)/w

    return [xcenter,ycenter,wd,hd]

class CustomDetections(sv.Detections):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.class_name:list = []

for file in os.listdir(VIDEO_ROOT):
# model = YOLO('yolov8x-seg.pt').cuda()
    filename = file.rsplit(".",1)[0]
    frame_generator = sv.get_video_frames_generator(source_path=os.path.join(VIDEO_ROOT,file))
    byte_tracker = sv.ByteTrack()

    current_dataroot = os.path.join(DATAROOT,filename)
    os.makedirs(current_dataroot,exist_ok=True)
    for n,frame in enumerate(tqdm(frame_generator)):
        
        name = f"{filename}_00{n}"
        # segmentation info is created
        if n % 15 != 0:
            continue
        h,w,_ = frame.shape
        result = model.track(frame, imgsz = 320,verbose=False)[0]
        detections = CustomDetections.from_ultralytics(result)
        if len(detections) == 0:
            continue
        towrite = np.array([calculate_cords(detection[0],h,w) for detection in detections])

        towrite = towrite.tolist()
        #convert list of number in string
        pred2write = [list(map(str,values)) for values in towrite]
        file2write = ["0 "+" ".join(pred) for pred in pred2write]

        towrite = "\n".join(file2write)

        # detections = byte_tracker.update_with_detections(detections)
        cv2.imwrite(os.path.join(current_dataroot,name+".png"),frame)
        with open(os.path.join(current_dataroot,name+".txt"),"w") as file:
            file.writelines(towrite)
            pass
            

        # segmentation info is no longer heres
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections)

        # cv2.imshow("frame",cv2.resize(annotated_frame,(0,0),fx=0.5,fy=0.5))
        cv2.imshow("frame",annotated_frame)
        cv2.waitKey(1)
