'''
Created on Jul. 17, 2019

    create densecrf testing
    note:
        pip install pydensecrf
    
@author: deisler
'''
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import glob
import cv2 as cv
import os


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q
    
# Setting of CRF parameters
def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor

imagepath = os.path.join(os.getcwd(),'image')
maskpath = os.path.join(os.getcwd(),'mask')

def get_fileslist( imagepath = imagepath ):
    print(os.path.join(imagepath, '*'))
    return glob.glob( os.path.join(imagepath, '*'))

def get_imagename( imagepath ):
    
    return imagepath.split('/')[-1]

def check_image(src):
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return False
    else:
        return True
    
def get_colormask( image ):
    if len(image.shape) == 3:
        mask = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        mask = image.copy()
#     thresh = cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    retval, thresh = cv.threshold(mask,0.1, 255,cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(
       thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    mask_img = np.zeros((mask.shape[0],mask.shape[1],3), np.uint8)
    mask_img = cv.drawContours(mask_img, contours, -1, (0,220,0), -1)
    
#     if debug_mode:
#         cv.imshow('mask', mask)
#         cv.imshow('thresh', thresh)
#         cv.imshow('maskimg', mask_img)
#         cv.waitKey(0)
        
    return mask_img

def mask_on_image(imagepath, maskpath):
    
    img = cv.imread(imagepath)
    mask = cv.imread(maskpath)
    
    if check_image( img ) and check_image( mask ):
        mask = get_colormask( mask )
        
    maskimage = np.where(mask, img*0.3 + mask*0.7, img).astype(np.uint8)
    
#     if debug_mode:
#         cv.imshow('maskimage', maskimage)
#         cv.waitKey(0)
        
    return maskimage

        
def generate_crf_mask(maskpath):
    mask = cv.imread(maskpath)
    if check_image(mask):
        if len(mask.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # normalize mask with NORM_MINMAX, nORM_L1 or NORM_L2
        norm_mask = cv.normalize(mask, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)    
        
        norm_blur_mask = cv.GaussianBlur(norm_mask,(7,7),0)
        retval, thresh_mask = cv.threshold(norm_blur_mask,0.1, 1,cv.THRESH_BINARY)
        
        #crf_mask shape (label,height,width)
        crf_mask=np.zeros((thresh_mask.shape[0],thresh_mask.shape[1],2))
        crf_mask[:,:,0] = (1-thresh_mask[:,:]) 
        crf_mask[:,:,1] = thresh_mask[:,:]
        crf_mask = crf_mask.transpose(2, 0, 1)
        
        print('crf_mask shape:',crf_mask.shape)
        
        return crf_mask
    else:
        return None 
    
def crf_test():
    crf_demo = DenseCRF(10,3,3,3,80,13)
    
    imagelist = get_fileslist(imagepath)
    masklist = get_fileslist(maskpath)
    
    print( imagelist[1],masklist[1])
    
    for imgpath in imagelist:
        imagename = get_imagename(imgpath)
        maskname = imagename[:-3] + 'png'
        mkpath = os.path.join(maskpath, maskname)
        
        img = cv.imread(imgpath)
        crf_mask = generate_crf_mask(mkpath)
        
        if crf_demo and img is not None:
            if len(crf_mask.shape) < 3:
                crf_mask = np.expand_dims(crf_mask, -1)
                print('crf_mask expand_shape:',crf_mask.shape)
            crf_mask = crf_demo(img, crf_mask)
            crf_mask = np.argmax(crf_mask, axis=0)
            crfmaskpath = os.path.join(maskpath, imagename[:-4]+'_crfmask.jpg')
            if len(crf_mask.shape) < 3:
                crf_mask = np.expand_dims(crf_mask, -1)
                print('crf_mask expand_shape:',crf_mask.shape)
            cv.imwrite(crfmaskpath, crf_mask)
            print(crf_mask.shape)

            crfmask_on_img = mask_on_image(imgpath,crfmaskpath)
            blendimgpath = os.path.join(maskpath, imagename[:-4]+'_crfimg.jpg')
    
            cv.imwrite(blendimgpath, crfmask_on_img)
    

def main():
    
    crf_test()

if __name__ == "__main__":
    main()
    
    
