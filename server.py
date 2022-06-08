import cv2
import numpy as np
from multiprocessing import shared_memory,Process
import time
from yolov5 import run
from pre import cam
import time 

BUF_SZ = 10
NUM_PROC = 2
NUM_DETS = 300
SOURCE = 0
MODEL = "yolov5m_leaky_352_0mean_uint8.tmfile"

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

class Khadas():
    def __init__(self):
        self.cam = Process(target=cam, args = (SOURCE,))
        self.cam.start()
        time.sleep(1)    
        self.ex_stop = shared_memory.SharedMemory(name = "stop") 
        self.stop =  np.ndarray([1], dtype=np.uint8, buffer=self.ex_stop.buf)  
        self.last_model = MODEL
        self.upload_models(MODEL)
        self.ex_raw = shared_memory.SharedMemory(name="raw-frame")
        self.ex_dets = shared_memory.SharedMemory(name = "dets")
        self.ex_status = shared_memory.SharedMemory(name="status") 
        self.ex_counter = shared_memory.SharedMemory(name="counter") 
        self.raw =  np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=self.ex_raw.buf)	 
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.counter = np.ndarray([1], dtype=np.int64, buffer=self.ex_counter.buf)
        self.begin = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.max_fps = 0
        self.thresh = 20
        self.frame = None
        self.dets = None

    def upload_models(self, model, change = False):
        self.stop[0] = 1
        if change:
            self.m1.kill()
            self.m2.kill()
        self.m1 = Process(target=run, args = (0, model))
        self.m1.start()
        self.m2 = Process(target=run, args = (1, model))
        self.m2.start()
        if change:
            time.sleep(5)  # wait so that model can start     
            self.m1.join(timeout=0)
            self.m2.join(timeout=0)
            if not self.m1.is_alive() or not self.m2.is_alive():
                self.stop[0] = 1
                self.m1.kill()
                self.m2.kill()
                self.m1 = Process(target=run, args = (0, self.last_model))
                self.m1.start()
                self.m2 = Process(target=run, args = (1, self.last_model))
                self.m2.start()
                print("Corrupted model!")
            else:
                self.last_model = model  

    def show(self):
        tail = self.counter[0]%BUF_SZ
        idx = np.where(self.status==3)[0]
        queue = np.concatenate((idx[idx>tail], idx[idx<tail]))
        
        if len(queue):
            if len(queue)>2:
                for i in range(len(queue)-2):
                    self.status[queue[i]] = 0
            
            elif not self.status[queue[-1] - 1]  == 2: # checking if penultimate image is being inferenced
                for idx in queue:
                    self.status[idx] = 0                   
                    self.frame_counter+=1
                    frame = self.raw[idx]
                    self.dets = self.dets_buf[idx]
                    self.frame = self.post(frame, self.dets)
                    if not self.frame_counter % 30:
                        self.fps = 30/(time.time() - self.begin)
                        if self.fps > self.max_fps:
                            self.max_fps = self.fps
                        self.begin = time.time()
                    cv2.imshow('frame', self.frame)
                    key = cv2.waitKey(1)
            print("Max FPS %.2f, Current Fps: %.2f"%(self.max_fps, self.fps), end = "\r")

    def post(self, frame, dets):
        for det in dets:
            if det[5] == 0:
                return frame
            if det[5] < self.thresh:
                continue
            b = 100 + (25 * det[4]) % 156
            g = 100 + (80 + 40 * det[4]) % 156
            r = 100 + (120 + 60 * det[4]) % 156
            color = (b, g, r)
            start = (int(det[0]), int(det[1])) 
            end =(int(det[0]+det[2]), int(det[1] + det[3])) 
            cv2.rectangle(frame,start, end, color, 2)
            text = "%.0f "%(det[5]) + names[int(det[4])]
            cv2.putText(frame,text,
                        (start[0]+5, start[1]-5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        color, 1)
        return frame

    async def get_frame(self):
        while self.frame is None:
            pass
        return self.frame
    
if __name__ == "__main__":

    khadas = Khadas()

    start = time.time()
    while True:
        khadas.show()
        # if 35 > (time.time() - start) > 30:
        #     print("Changing model")
        #     khadas.upload_models("pre.py", change = True)
        
        
    
