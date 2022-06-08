from pickle import FALSE
import cv2
import numpy as np
from multiprocessing import shared_memory,Process
import time
import time 
import copy
from ctypes import *
BUF_SZ = 10
NUM_PROC = 2
NUM_DETS = 300
SOURCE = 0
MODEL = "yolov5m_leaky_352_0mean_uint8.tmfile"
VID = False # change if you want to inference on video
FPS = 30    # set fps for video 

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


libc = cdll.LoadLibrary("./yolov5_lib.so")
libc.set_image_wrapper.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int]
libc.postpress_graph_image_wrapper.argtypes = [c_int, c_int, c_void_p, c_void_p, c_int,
                                               c_int, c_int, c_int, c_int, c_float, c_float]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    im = im[::,::,::-1].transpose((2, 0, 1))
    return im


def cam(source):
    shm_pre = shared_memory.SharedMemory(create=True, size=BUF_SZ*3*352*352, name = "pre-frame") 
    shm_raw = shared_memory.SharedMemory(create=True, size=BUF_SZ*3*480*640, name = "raw-frame") 
    shm_thresh = shared_memory.SharedMemory(create=True, size=4, name = "thresh") 
    shm_nms = shared_memory.SharedMemory(create=True, size=4, name = "nms") 
    shm_counter = shared_memory.SharedMemory(create=True, size=8, name = "counter") 
    shm_status = shared_memory.SharedMemory(create=True, size=BUF_SZ, name = "status") 
    shm_dets = shared_memory.SharedMemory(create=True, size=BUF_SZ*4*NUM_DETS*6, name = "dets") 
    shm_stop = shared_memory.SharedMemory(create=True, size=1, name = "stop") 
    thresh =  np.ndarray([1], dtype=np.float32, buffer=shm_thresh.buf)
    thresh[0] = 0.2
    nms =  np.ndarray([1], dtype=np.float32, buffer=shm_nms.buf)
    nms[0] = 0.2
    stop =  np.ndarray([1], dtype=np.uint8, buffer=shm_stop.buf)
    stop[0] = 1 
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)	
    pre = np.ndarray([BUF_SZ, 3, 352, 352], dtype=np.uint8, buffer=shm_pre.buf)
    raw = np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=shm_raw.buf)
    counter = np.ndarray([1], dtype=np.int64, buffer=shm_counter.buf)
    status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=shm_status.buf)
    counter[0] = 0
    try:
        while True:	
            if not stop:	
                start_main = time.time()	
                ret, frame = cap.read()
                if VID:
                    frame = cv2.resize(frame, [640,480], interpolation = cv2.INTER_LINEAR)
                start = time.time()
                raw[counter[0]%BUF_SZ][:] = frame 
                pre[counter[0]%BUF_SZ][:] = letterbox(frame, new_shape=(352, 352))
                status[counter[0]%BUF_SZ] = 1
                counter[0] += 1	
                if VID:
                    slp = 1/FPS -(time.time() - start_main)
                    if slp > 0:
                        time.sleep(slp)
                # print("Camera FPS: ", 1/(time.time() - start_main))

    finally:
        print("\nDeleting object")
        shm_counter.close()
        shm_counter.unlink()
        shm_pre.close()
        shm_pre.unlink()
        shm_raw.close()
        shm_raw.unlink()
        shm_status.close()
        shm_status.unlink()

class YOLOV5():
    def __init__(self, proc, model):
        self.proc = proc
        print("proc: ", self.proc)
        self.img_sz = 352
        self.context = libc.create_context("timvx".encode('utf-8'), 1)
        libc.init_tengine()
        libc.set_context_device(self.context, "TIMVX".encode('utf-8'), None, 0)
        model_file = model.encode('utf-8')
        self.graph = libc.create_graph(self.context, "tengine".encode('utf-8'), model_file)
        libc.set_graph(352 , self.img_sz , self.graph)
        self.input_tensor = libc.get_graph_input_tensor(self.graph, 0, 0)
        self.output_node_num = libc.get_graph_output_node_number(self.graph)
        self.dets = np.zeros([NUM_DETS,6], dtype = np.float32)  
        self.classes = libc.get_classes(self.graph)
        self.last_dets = None
        self.current = -1
        print("Initialised")
        self.ex_pre = shared_memory.SharedMemory(name="pre-frame") #preprocessed images
        self.ex_counter = shared_memory.SharedMemory(name="counter") # number of images read from camera
        self.ex_status = shared_memory.SharedMemory(name="status") # not inferenced images
        self.ex_dets = shared_memory.SharedMemory(name = "dets") # array of detections
        self.ex_stop = shared_memory.SharedMemory(name = "stop") # array of detections
        self.ex_thresh = shared_memory.SharedMemory(name = "thresh") 
        self.ex_nms = shared_memory.SharedMemory(name = "nms") 
        self.pre = np.ndarray([BUF_SZ, 3, 352, 352], dtype=np.uint8, buffer=self.ex_pre.buf)
        self.counter = np.ndarray([1], dtype=np.int64, buffer=self.ex_counter.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.stop = np.ndarray([1], dtype=np.uint8, buffer=self.ex_stop.buf)
        self.thresh = np.ndarray([1], dtype=np.float32, buffer=self.ex_thresh.buf)[0] # get thresh from buffer
        self.nms = np.ndarray([1], dtype=np.float32, buffer=self.ex_nms.buf)[0] # get iou thresh from buffer
        self.stop[0] = 0

    def inference(self):
        if self.status[(self.counter[0]-1)%BUF_SZ] and (self.counter[0]-1)%NUM_PROC == self.proc:
            self.current = (self.counter[0]-1)
            self.status[self.current%BUF_SZ] = 2 # start inference
            self.frame = self.pre[self.current%BUF_SZ]
            libc.set_image_wrapper(self.frame.ctypes.data, 352, 352, self.input_tensor, self.img_sz, self.img_sz )
            libc.run_graph(self.graph, 1)
            libc.postpress_graph_image_wrapper(480, 640, self.dets.ctypes.data, 
                                                self.graph,self.output_node_num, 352 ,self.img_sz, self.classes, NUM_DETS, self.nms, self.thresh)
            self.last_dets = copy.copy(self.dets)
            self.dets_buf[self.current%BUF_SZ][:] = self.dets[:]
            self.status[self.current%BUF_SZ] = 3 # inferenced

def run(proc, model):
    yolov5 = YOLOV5(proc, model)	
    while True:
        yolov5.inference()

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
        self.ex_thresh = shared_memory.SharedMemory(name = "thresh")
        self.ex_nms = shared_memory.SharedMemory(name = "nms")
        self.raw =  np.ndarray([BUF_SZ, 480, 640, 3], dtype=np.uint8, buffer=self.ex_raw.buf)	 
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.counter = np.ndarray([1], dtype=np.int64, buffer=self.ex_counter.buf)
        self.thresh = np.ndarray([1], dtype=np.float32, buffer=self.ex_thresh.buf)
        self.nms = np.ndarray([1], dtype=np.float32, buffer=self.ex_nms.buf)
        self.begin = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.max_fps = 0
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
        
        
    
