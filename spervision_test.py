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

    

timer = Timer()


#VIDEO_ROOT = "./walking.mp4" #"./walking_people.mp4"
VIDEO_ROOT = "./walking_people.mp4"

if __name__ == "__main__":
    #model = YOLO('yolov8n-seg.pt').cuda()
    model = YOLO("yolov8x.pt").cuda()  

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator(thickness=1,
                                text_scale=0.5,
                                text_padding=5)
    track_annotator = sv.TraceAnnotator(trace_length=100)

    video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_ROOT)
    frame_generator = sv.get_video_frames_generator(source_path=VIDEO_ROOT)
    byte_tracker = sv.ByteTrack(track_buffer=100)

    with sv.VideoSink(f'test_box_and_mask_tracking_notkeepall.mp4', video_info=video_info) as sink:
        i = 0
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            
            result = model.track(frame, imgsz = 1920,verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            before_lenght = len(detections)
            prova = detections
            detections = byte_tracker.update_with_detections(detections,keep_all=True)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            annotated_frame = track_annotator.annotate(
                scene=annotated_frame.copy(),
                detections=detections)
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame.copy(),
                detections=detections)
            

            sink.write_frame(frame=annotated_frame)

            cv2.imshow("frame",cv2.resize(annotated_frame,(0,0),fx=0.5,fy=0.5))
            cv2.waitKey(1)
            

            i+=1
            # if i == 50:
            #     break

        #timer.get_statistics()
                        an_frame = track_annotator.annotate(
                scene=an_frame.copy(),
                detections=tracked_det
            )

            vsink.write_frame(frame=an_frame)
