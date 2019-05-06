'''
Created on Feb. 26, 2019

    create data_name file and augment image by scale, flip, and crop

@author: deisler
'''

import glob
import os
import cv2
import numpy as np
import math

root = '/notebooks/maskrcnn-benchmark/datasets/coco/Matting_data/'

train_image_size = 800.0

image_path = '/notebooks/maskrcnn-benchmark/datasets/coco/Matting_data/fg/'
imagelist = glob.glob(os.path.join(image_path, '*'))
imagelist.sort()

image_fg = [ image for image in imagelist if "copy" not in image ]
print(len(image_fg))

with open('/notebooks/maskrcnn-benchmark/datasets/coco/Matting_data/training_fg_names.txt', 'w') as f:
    for item in image_fg:
        item = item.split('/')[-1]
        f.write("%s\n" % item)
        
with open('/notebooks/maskrcnn-benchmark/datasets/coco/Matting_data/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()


image_bg_path = os.path.join(root,'coco2017_no_people') 
image_bg_list = glob.glob(os.path.join(image_bg_path,'*'))

image_bg_list = np.random.choice(image_bg_list,25000,False)

print(image_bg_list[:5])

with open('/notebooks/maskrcnn-benchmark/datasets/coco/Matting_data/training_bg_names.txt', 'w') as f:
    for item in image_bg_list:
        item = item.split('/')[-1]
        f.write("%s\n" % item)


def random_crop_fg( image ):
    wratio = np.random.uniform(0.5,1)
    xstart_ratio = np.random.uniform(0, 1-wratio)
    wratio = round(wratio,2)
    xstart_ratio = round(xstart_ratio, 2)
    hratio = np.random.uniform(0.5,1)
    ystart_ratio = np.random.uniform(0, 1-hratio)
    hratio = round(hratio,2)
    ystart_ratio = round(ystart_ratio,2)
    
    h,w = image.shape[:2]
    x = round( w * xstart_ratio )
    y = round( h * ystart_ratio ) 
    crop_height = round( h * hratio )
    crop_width = round( w * wratio )
    if len(image.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = image[y:y + crop_height, x:x + crop_width]
    return crop

def pre_flip(image):
    r = np.random.uniform(0,1)
    if r > 0.75:
        return np.fliplr(image)
    elif r > 0.5:
        return np.flipud( image )
    elif r > 0.25:
        image = np.fliplr(image)
        return np.flipud( image)
    else:
        return image

def pre_resize(image):
    h, w = image.shape[:2]
    maxvalue = max(h,w)
    if maxvalue > train_image_size:
        if h == maxvalue:
            r = train_image_size / h
            dim = ( int(image.shape[1] * r), int(train_image_size))
            print(dim)
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
        else:
            r = train_image_size / w
            dim = ( int(train_image_size), int(image.shape[0] * r ))
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
        return resized
    return image

def pre_scale(image):
    ratio = np.random.uniform(0.5, 0.9)
    res = cv2.resize(image,None,fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
    return res

def composite_fg_with_bg(im_name, bg_name):
    
    image = cv2.imread(fg_path + im_name, cv2.IMREAD_UNCHANGED)
#     print(fg_path + im_name)
#     print('image shape:',image.shape)
    image = pre_resize(image)
    
    if np.random.uniform(0,1) > 0.7:
        image = pre_scale(image)
        
    if np.random.uniform(0,1) > 0.5:
        image = pre_flip(image)
        
    if np.random.uniform(0.1) > 0.5:
        image = random_crop_fg(image)
#     print('image shape:',image.shape)
    b,g,r,a = cv2.split(image)

    im = np.stack((b,g,r),axis=-1)
#     a = cv.imread(a_path + im_name, 0)

    h, w = im.shape[:2]
    bg = cv2.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)
    
    print(im.shape,a.shape,bg.shape)
    
    
    return composite4(im, bg, a, w, h)

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


with open(os.path.join(root,'training_fg_names.txt')) as f:
    fg_files = f.read().splitlines()

with open(os.path.join(root,'training_bg_names.txt')) as f:
    bg_files = f.read().splitlines()

fg_path = os.path.join(root,'fg/')
bg_path = os.path.join(root,'coco2017_no_people/')
train_path = os.path.join(root,'train/')
alpha_path = os.path.join(root,'alpha/')
for i in range(10):
    for j in range(6):#len(fg_files)-500
        bg_file = np.random.choice(bg_files,1)
        print(fg_path,fg_files[i],bg_file[0])
        ret,a,_,_ = composite_fg_with_bg(fg_files[i],bg_file[0])
        imagename = fg_files[i].split('.')[0]+'-{}.'.format(j)+fg_files[i].split('.')[1]
        print(ret.shape,imagename)
        cv2.imwrite(os.path.join(train_path,imagename),ret)
        cv2.imwrite(os.path.join(alpha_path,'alpha-'+imagename),a)

