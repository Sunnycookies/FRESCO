import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str)
args = parser.parse_args()
root = args.root
img_root = os.path.join(root,'keys')

fps = 24    #FPS
# size=(640,512)    #图片、视频尺寸
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_root = os.path.join(root,'')+'video24.mp4'

file_list = os.listdir(img_root)
file_list.sort()
ref_frame = cv2.imread(os.path.join(img_root, file_list[0]))
(H, W, C) = ref_frame.shape
size = (W, H)

videoWriter = cv2.VideoWriter(video_root,fourcc,fps,size, True)

for img_file in file_list:
    frame = cv2.imread(os.path.join(img_root, img_file))
    videoWriter.write(frame)
print('Done!\n')
    
videoWriter.release()