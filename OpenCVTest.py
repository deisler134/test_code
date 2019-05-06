'''
Created on Jan. 31, 2019

@author: deisler
'''
import cv2 as cv
import numpy as np
import imutils

videopath = '/home/deisler/Downloads/resource/project/maskrcnn-benchmark/testvideo/VID_20190131_110520.mp4'



def show_video(videopath, is_rotate = False, angle = 90, is_save = False):
    
    cam = cv.VideoCapture(videopath)
    key = 10
    
    if is_save:
        save_path = videopath[:-4] + '_save'  + videopath[-4:]
        
        print(save_path)
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        
        print(cam.get(cv.CAP_PROP_FRAME_WIDTH),cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        if is_rotate:
            saver = cv.VideoWriter(save_path, fourcc, 20, (int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)),int(cam.get(cv.CAP_PROP_FRAME_WIDTH))))
        else:
            saver = cv.VideoWriter(save_path, fourcc, 20, (int(cam.get(cv.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))))
        
    
    
    while(cam.isOpened()):
        ret, frame = cam.read()
        
        if is_rotate :
            frame = imutils.rotate_bound(frame, angle)
#             print(frame.shape)
        if is_save:
            saver.write(frame)

        cv.imshow('frame',frame)
        ret = cv.waitKey( key )
        if ret == ord('q'):
            break
        elif ret == ord('p'):
            key = 0
        elif ret == ord('c'):
            key = 10
    cam.release()
    if is_save:
        saver.release()
    cv.destroyAllWindows()  

show_video(videopath, True, 90, True) 


  