'''
Created on Mar. 12, 2019

    load coco json file to check annotations
    
@author: deisler
'''

import json
import numpy as np
import image_data_preprocessor


filepath = '/media/deisler/Data/project/coco/cocodata/annotations2017/instances_train2017.json'

from pprint import pprint

with open(filepath) as f:
    data = json.load(f)
    
    for image in data['images']:
        if image['file_name'] == '000000000036.jpg':
            print(image)
            print(image['id'])
            print(data['annotations'][5])
            break
    for ann in data['annotations']:
#         print(ann['id'])
        if ann['iscrowd']:
            print(ann)
            break
    for image in data['images']:
        if image['id'] == 153344:
            print(image)
            print(image['id'])
            break
    for ann in data['annotations']:
#         print(ann['id'])
        if ann['image_id'] == 36:
            print(ann)
            break

#             break
#             break
#     print(data)

# pprint(data)
    
def prefilter(self, dataset):
    res_annos = []
    annos = dataset.dataset['annotations']
    for anno in annos:
        # throw away all crowd annotations
        if anno['iscrowd']: continue
        # filter no person annotation
        if anno['category_id'] != 1: continue

        m = dataset.annToMask(anno)
        mask_area = np.count_nonzero(m)
        if mask_area / float(m.shape[0] * m.shape[1]) > self.fg_thresh:
            anno['bbox'] = get_mask_bbox(m)
            res_annos.append(anno)
    return res_annos

