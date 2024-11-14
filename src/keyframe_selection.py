import cv2
import torch.nn.functional as F
import numpy as np
from src.utils import *

def insert_key(keys, ind):
    for i, k in enumerate(keys):
        if ind < k:
            keys.insert(i, ind)
            break
            
def get_maxinterv(keys):
    maxinterv = 1
    for i in range(len(keys)-1):
        tmp = keys[i+1]-keys[i]
        if tmp > maxinterv:
            maxinterv = tmp
    return maxinterv

def get_keyframe_ind(filename, lastframen = 1e10, mininterv = 5, maxinterv = 20, viz = False):
    video_cap = cv2.VideoCapture(filename)
    n_frames = max(1, min(int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)), lastframen))
    
    if maxinterv == mininterv:
        return list(range(0, n_frames, mininterv))
    err = [0]
    preframe = None
    for i in range(n_frames):
        success, frame = video_cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        img = cv2.GaussianBlur(img, (9, 9), 0.0)
        if i == 0:
            preframe = numpy2tensor(img)
        else:
            curframe = numpy2tensor(img)
            err += [float(F.mse_loss(preframe, curframe).cpu().numpy())]
            preframe = curframe
    err = np.array(err)
    err1 = np.array(err)
    
    n_frames = len(err)
    keys = [0, n_frames-1]
    err[0:mininterv] = -1
    err[-mininterv:] = -1

    while get_maxinterv(keys) > maxinterv:
        ind = np.argmax(err)
        if err[ind] == -1:
            break
        err[ind-mininterv:ind+mininterv] = -1
        insert_key(keys, ind) 
    
    if viz:
        plt.plot(err1)
        plt.plot(keys, err1[keys], 'bo')
        plt.show()
    
    return keys

def get_keyframe_ind_extend(filename, mode = 'fixed', radix = 1, indexes = [], 
                            mininterv = 5, maxinterv = 20, lastframen = 1e10, viz = False):
    if mode not in ['fixed', 'loop']:
        print(f"keyframe selection mode {mode} not implemented")
        raise NotImplementedError
    
    # loop
    if mode == 'loop':
        radix = min((len(indexes) - 1 // 2, radix))
        return [[indexes[0]] + [indexes[j] for j in range(i, len(indexes), radix)] for i in range(1, radix + 1)]
    
    # fixed
    if mininterv == maxinterv:
        return [[indexes[i] for i in range(0, len(indexes), radix)], ]
    
    video_cap = cv2.VideoCapture(filename)
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = max(1, min(n_frames, lastframen))

    err = [0]
    preframe = None
    for i in range(n_frames):
        success, frame = video_cap.read()
        if not success:
            break
        if i not in indexes:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        img = cv2.GaussianBlur(img, (9, 9), 0.0)
        if i == 0:
            preframe = numpy2tensor(img)
        else:
            curframe = numpy2tensor(img)
            err += [float(F.mse_loss(preframe, curframe).cpu().numpy())]
            preframe = curframe
    err = np.array(err)
    err1 = np.array(err)
    
    n_frames = len(err)
    keys = [0, n_frames-1]
    err[0:mininterv] = -1
    err[-mininterv:] = -1

    while get_maxinterv(keys) > maxinterv:
        ind = np.argmax(err)
        if err[ind] == -1:
            break
        err[ind-mininterv:ind+mininterv] = -1
        insert_key(keys, ind) 
    
    if viz:
        plt.plot(err1)
        plt.plot(keys, err1[keys], 'bo')
        plt.show()
    
    return [[indexes[k] for k in keys], ]
        