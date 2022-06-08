import cv2
import numpy as np
from multiprocessing import shared_memory
import time

BUF_SZ = 10
NUM_PROC = 2
NUM_DETS = 300

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

class Img_buffer():
    def __init__(self):      
        self.ex_raw = shared_memory.SharedMemory(name="raw-frame")
        self.ex_read = shared_memory.SharedMemory(name="read")
        self.ex_dets = shared_memory.SharedMemory(name = "dets")
        self.ex_status = shared_memory.SharedMemory(name="status") # not inferenced images = 1
        self.raw =  np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=self.ex_raw.buf)
        self.read = np.ndarray([NUM_PROC], dtype=np.int64, buffer=self.ex_read.buf)		 
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.counter = [-1]*NUM_PROC
        self.begin = time.time()
        self.frame_counter = 0
        self.proc = 0
        self.last = np.amin(self.read)
        self.temp = 0
        self.fps = 0
        self.max_fps = 0
        self.frame = None
        self.dets = None

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
            print("Max FPS %.2f, Current Fps: %.2f"%(self.max_fps, self.fps))

    def post(self, frame, dets):
        for det in dets:
            if det[5] == 0:
                return frame
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

if __name__ == "__main__":
    buf = Img_buffer()
    while True:
        buf.show()
    
