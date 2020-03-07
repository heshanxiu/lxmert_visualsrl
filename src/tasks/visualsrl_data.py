# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 512

# The path to data and image features.
# VQA_DATA_ROOT = 'data/visualsrl/'
VISUALSRL_IMGFEAT_ROOT = 'data/visualsrl_imgfeat/'
SPLIT2NAME = {
    'train': 'train',
    'dev': 'dev',
    'minidev': 'tiny_dev',
    'minitrain': 'tiny_train',
    'test': 'test',
}
SPLIT2NUM = {
    'train': 22056,
    'dev': 15656,
    'minidev': 100,
    'minitrain': 200,
    'test': 25196,
}

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VisualSRLTorchDataset(Dataset):
    def __init__(self, splits: str):
        super().__init__()

        self.splits = splits

        # Loading detection features to img_data
        self.img_data = []
        # for split in self.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
        load_topk = SPLIT2NUM[splits]
        self.img_data.extend(load_obj_tsv(
            os.path.join(VISUALSRL_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[self.splits])),topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:
            self.imgid2img[img_datum['img_id']] = img_datum


        print("Use %d data in torch dataset" % (len(self.img_data)))
        print()

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item: int):

        # Get image info
        img_info = self.img_data[item]
        img_name = img_info['img_id']
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return img_name, feats, boxes

