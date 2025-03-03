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

model = YOLO(r'C:\Users\anton\Desktop\prova\runs\detect\train3\weights\best.pt').cuda()           
line_counter = LineCounter([0,110])

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(thickness=1,
                                text_scale=0.2,
                                text_padding=4)
track_annotator = sv.TraceAnnotator()
timer = Timer()

VIDEO_ROOT = r"C:\Users\anton\Desktop\VIDEO_ANNOTATION"

class CustomDetections(sv.Detections):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.class_name:list = []

for file in os.listdir(VIDEO_ROOT):
# model = YOLO('yolov8x-seg.pt').cuda()
    filename = file.rsplit(".",1)[0]
    video_info = sv.VideoInfo.from_video_path(video_path=os.path.join(VIDEO_ROOT,file))
    frame_generator = sv.get_video_frames_generator(source_path=os.path.join(VIDEO_ROOT,file))
    byte_tracker = sv.ByteTrack()

    with sv.VideoSink(f'RESULT_{filename}.mp4', video_info=video_info) as sink:
        i = 0
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            
            # segmentation info is created
            result = model.track(frame, imgsz = 320,verbose=False)[0]
            detections = CustomDetections.from_ultralytics(result)
            detections = byte_tracker.update_with_detections(detections)

            # segmentation info is no longer heres
            timer.start()
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            timer.end()
            annotated_frame = track_annotator.annotate(
                scene=annotated_frame.copy(),
                detections=detections)
            
            flatten_annotated = annotated_frame.reshape((320*240,3))
            plottempalte = TEMPLATE.copy()
            plottempalte[TEMPLATE_IDX] = flatten_annotated[TEMPLATE_IDX]
            annotated_frame = plottempalte.reshape((240,320,3))
            line_counter.check_pass(detections=detections,annotator=track_annotator)
            cv2.line(annotated_frame,(0,line_counter.y),(319,line_counter.y),(0,255,0))
            cv2.line(annotated_frame,(0,line_counter.y_out),(319,line_counter.y_out),(0,0,255))
            cv2.putText(annotated_frame,f"out: {line_counter.counter_out}",(0,26),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.putText(annotated_frame,f"in:  {line_counter.counter_in}",(1,68),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            # cv2.imshow("frame",cv2.resize(annotated_frame,(0,0),fx=0.5,fy=0.5))
            cv2.imshow("frame",annotated_frame)
            cv2.waitKey(1)
            

            sink.write_frame(frame=annotated_frame)

        timer.get_statistics()