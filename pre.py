import cv2
import time 
import numpy as np
from multiprocessing import shared_memory

SOURCE = 0
BUF_SZ = 10
FPS = 30 # set fps for video
VID = False # change if you want to run video
NUM_PROC = 2
NUM_DETS = 300

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  # only scale down, do not scale up (for better val mAP)
    #     r = min(r, 1.0)

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
    shm_counter = shared_memory.SharedMemory(create=True, size=8, name = "counter") 
    shm_status = shared_memory.SharedMemory(create=True, size=BUF_SZ, name = "status") 
    shm_dets = shared_memory.SharedMemory(create=True, size=BUF_SZ*4*NUM_DETS*6, name = "dets") 
    shm_stop = shared_memory.SharedMemory(create=True, size=1, name = "stop") 
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
                    print("ONLYFORVIDEO")
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

if __name__ == "__main__":
    try:

        ps = mp.Process(target=cam, args = (), daemon = True)
        ps.start()
        time.sleep(0.5)

        while True:
            time.sleep(0.05)
    finally:
        print("\nClosing")



