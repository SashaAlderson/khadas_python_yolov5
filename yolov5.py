from ctypes import *
from multiprocessing import shared_memory
import time 
import copy
import numpy as np
import threading
from multiprocessing import shared_memory, Process
from post import Img_buffer
from pre import *

BUF_SZ = 10
NUM_PROC = 2
PROC = 0
CLASSES = 80
NUM_DETS = 300
SOURCE = 0 
MODEL = "yolov5m_leaky_352_0mean_uint8.tmfile"
libc = cdll.LoadLibrary("./yolov5_lib.so")
libc.set_image_wrapper.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int]
libc.postpress_graph_image_wrapper.argtypes = [c_int, c_int, c_void_p, c_void_p,
                                                c_int , c_int, c_int, c_int, c_int, c_float]

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
        self.nms = 0.2
        self.current = -1
        print("Initialised")
        self.ex_pre = shared_memory.SharedMemory(name="pre-frame") #preprocessed images
        self.ex_counter = shared_memory.SharedMemory(name="counter") # number of images read from camera
        self.ex_status = shared_memory.SharedMemory(name="status") # not inferenced images
        self.ex_dets = shared_memory.SharedMemory(name = "dets") # array of detections
        self.ex_stop = shared_memory.SharedMemory(name = "stop") # array of detections
        self.pre = np.ndarray([BUF_SZ, 3, 352, 352], dtype=np.uint8, buffer=self.ex_pre.buf)
        self.counter = np.ndarray([1], dtype=np.int64, buffer=self.ex_counter.buf)
        self.status = np.ndarray([BUF_SZ], dtype=np.uint8, buffer=self.ex_status.buf)
        self.dets_buf = np.ndarray([BUF_SZ, NUM_DETS, 6], dtype=np.float32, buffer=self.ex_dets.buf)
        self.stop = np.ndarray([1], dtype=np.uint8, buffer=self.ex_stop.buf)
        self.stop[0] = 0

    def inference(self):
        if self.status[(self.counter[0]-1)%BUF_SZ] and (self.counter[0]-1)%NUM_PROC == self.proc:
            self.current = (self.counter[0]-1)
            self.status[self.current%BUF_SZ] = 2 # start inference
            self.frame = self.pre[self.current%BUF_SZ]
            libc.set_image_wrapper(self.frame.ctypes.data, 352, 352, self.input_tensor, self.img_sz, self.img_sz )
            libc.run_graph(self.graph, 1)
            libc.postpress_graph_image_wrapper(480, 640, self.dets.ctypes.data, 
                                                self.graph,self.output_node_num, 352 ,self.img_sz, self.classes, NUM_DETS, self.nms)
            self.last_dets = copy.deepcopy(self.dets)
            self.dets_buf[self.current%BUF_SZ][:] = self.dets[:]
            self.status[self.current%BUF_SZ] = 3 # inferenced
            
def run(proc, model):
    yolov5 = YOLOV5(proc, model)	
    while True:
        yolov5.inference()

if __name__ == "__main__":
    p1 = Process(target=cam, args = (SOURCE,))
    p1.start()
    p2 = Process(target=run, args = (0, MODEL))
    p2.start()
    p3 = Process(target=run, args = (1, MODEL))
    p3.start()
    time.sleep(1)
    ex_stop = shared_memory.SharedMemory(name = "stop") 
    stop =  np.ndarray([1], dtype=np.uint8, buffer=ex_stop.buf)
    while True:
        command = input()
        if command == 's':
            stop[0] = 1

        if command == 'r':
            stop[0] = 0

        if command == 'c':
            stop[0] = 1
            p2.kill()
            p3.kill()
            p2 = Process(target=run, args = (0, MODEL), daemon = True)
            p2.start()
            p3 = Process(target=run, args = (1, MODEL), daemon = True)
            p3.start()
            