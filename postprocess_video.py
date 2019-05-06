'''
Created on Mar. 11, 2019

    postprocess video for image matting result

@author: deisler
'''
import cv2 as cv
import pandas
import matplotlib
import h5py
import numpy as np
import time

videopath = '/media/deisler/Data/project/coco/cocodata/Deep-Image-Matting/data/VID_20190131_111425_out.avi'


def video_save(videopath, w = 1920, h = 1080, is_rotate = False):
    
#     cam = cv.VideoCapture(videopath)
#     save_path = videopath[:-4] + '_save'  + videopath[-4:]
#     print(save_path)
#                 
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
        
#     width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
#     height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
#     
#     if width > w:
#         width = w
#     if height > h:
#         height = h
        
    if is_rotate:
        saver = cv.VideoWriter(videopath, fourcc, 20, (h, w))
    else:
        saver = cv.VideoWriter(videopath, fourcc, 20, (w, h))
        
    return saver

def generate_contour_matting(videopath):
    cam = cv.VideoCapture(videopath)
    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    
#     tri_videopath = videopath[:-4] + '_post.avi'
#     tri_video = video_save(tri_videopath, w = width, h = height)
    
    while(cam.isOpened()):
        start_time = time.time()
        ret, frame = cam.read()


#         if is_rotate :
#             frame = imutils.rotate_bound(frame, 90)
#             frame_mask = imutils.rotate_bound(frame_mask, 90)
        #             print(frame.shape)
        if frame is None:
            print ('Error image!')
            break
        
        print(frame.shape)
        
#         cv.GaussianBlur(frame,(3,3),0)
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        _,image_thresh = cv.threshold(frame,10, 255,cv.THRESH_BINARY)
        
        cv.imshow('thresh',image_thresh)
               
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        iter = 4    #np.random.randint(4, 8)
        fg = cv.dilate(image_thresh, kernel, iterations=iter)
        fg = cv.erode(fg, kernel, iterations=iter)
          
        cv.imshow('erode_dilate',fg)
#         
# #         cv.GaussianBlur(fg,(3,3),0)
#         
#         
        _, contours, hierarchy = cv.findContours(
            fg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)   #RETR_EXTERNAL
        mask_img = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        mask_img = cv.drawContours(mask_img, contours, -1, (220,220,220), -1)
     
        cv.imshow('out',frame)
        cv.imshow('post',mask_img)
        if cv.waitKey(40) & 0xff == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()
    
generate_contour_matting(videopath)    
    

    
    