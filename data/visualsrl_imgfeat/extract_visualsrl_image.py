# !/usr/bin/env python
# The root of bottom-up-attention repo. Do not need to change if using provided docker file.
BUTD_ROOT = '/opt/butd/'

# SPLIT to its folder name under IMG_ROOT
SPLIT2DIR = {
    'train': 'train',
    'dev': 'dev',
    'test': 'test',
}

import signal
import sys
import h5py


def signal_handler(sig, frame, name):
    print('This file causes core dump: ', name)
    exit(0)


import os, sys

sys.path.insert(0, BUTD_ROOT + "/tools")
os.environ['GLOG_minloglevel'] = '2'

import _init_paths
from examine_h5py import fix_index
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
from PIL import Image

import caffe
import argparse
import pprint
import base64
import numpy as np
import cv2
import csv
from tqdm import tqdm
import json

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

# Settings for the number of features per image. To re-create pre-trained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def load_image_ids(img_root, split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''

    signal.signal(signal.SIGQUIT, signal_handler)

    split = []
    if split_name == 'train':
        with open('train.json') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if i == 13820 or i == 14258 or i == 22786 or i == 22793 or i == 23119:
                    continue
                else:
                    dir_name = item.split('_')[0]
                    name = os.path.join(dir_name, item)
                    filepath = os.path.join(img_root, name)
                    split.append((filepath, item))
    elif split_name == 'dev':
        with open('dev.json') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if i == 2868 or i == 2867 or i == 15659:
                    continue
                dir_name = item.split('_')[0]
                name = os.path.join(dir_name, item)
                filepath = os.path.join(img_root, name)
                split.append((filepath, item))
    elif split_name == 'test':
        with open('test.json') as f:
            data = json.load(f)
            for item in data:
                dir_name = item.split('_')[0]
                name = os.path.join(dir_name, item)
                filepath = os.path.join(img_root, name)
                split.append((filepath, item))
    else:
        print('Unknown split')
    return split


def clean_up_download(img_ids):

    downloaded_img = []
    corrupted_index = []
    for i, img in enumerate(img_ids):

        img_path, img_id = img
        if os.path.isfile(img_path):
            try:
                curr_img = Image.open(img_path)
                curr_img.verify()
            except (IOError, SyntaxError) as e:
                if os.path.exists(img_path):
                    os.remove(img_path)
                corrupted_index.append(i)
                img_ids.remove(img)
        else:
            corrupted_index.append(i)
            img_ids.remove(img)

    print(corrupted_index)




def generate_tsv(prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    # never use set, it loses the order!!! F***
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['img_id'])
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print('already completed {:d}'.format(len(image_ids)))
    else:
        print('missing {:d}/{:d}'.format(len(missing), len(image_ids)))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for im_file, image_id in tqdm(image_ids):
                if image_id in missing:
                    try:
                        writer.writerow(get_detections_from_im(net, im_file, image_id))
                    except Exception as e:
                        print(e)


def generate_h5py(prototxt, weights, image_ids, h5py_file, index_file):

    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    image_to_index = {}
    num_found = 0

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            image_to_index = json.load(f)
            num_found = len(image_to_index.keys())
            found_ids = set(image_to_index.keys())

    if os.path.exists(h5py_file):
        pass
    else:
        with h5py.File(h5py_file, "a") as f:
            all_examples = len(wanted_ids)
            f.create_dataset("img_h", (all_examples,), dtype=np.float32)
            f.create_dataset('img_w', (all_examples,), dtype=np.float32)
            f.create_dataset('objects_id', (all_examples, 36), dtype=np.int64)
            f.create_dataset('objects_conf', (all_examples, 36), dtype=np.int64)
            f.create_dataset('attrs_id', (all_examples, 36), dtype=np.float32)
            f.create_dataset('attrs_conf', (all_examples, 36), dtype=np.float32)
            f.create_dataset('num_boxes', (all_examples,), dtype=np.float32)
            f.create_dataset('boxes', (all_examples, 36, 4), dtype=np.float32)
            f.create_dataset('features', (all_examples, 36, 2048), dtype=np.float32)

    missing = wanted_ids - found_ids
    # fixing indexes
    valid, actual_missing = fix_index(h5py_file, 'train.json', index_file)
    actual_missing = str(actual_missing)
    print("actual misssing {}".format(actual_missing))

    if len(missing) == 0:
        print('already completed {:d}'.format(len(image_ids)))
    else:
        print('missing {:d}/{:d}'.format(len(missing), len(image_ids)))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with h5py.File(h5py_file, "a") as f:
            for i, (im_file, image_id) in enumerate(tqdm(image_ids)):
                print("curr {}".format(image_id))
                if i == 23115 or i == 47545:
                    pass
                # if i == 15659: # dev set escape
                #     pass

                elif image_id in actual_missing:
                    try:
                        print("enter")
                        print("image id {}".format(image_id))
                        index = valid.pop(0)
                        img_dict = get_detections_from_im(net, im_file, image_id)
                        f["img_h"][index] = img_dict["img_h"]
                        f["img_w"][index] = img_dict["img_w"]
                        f["objects_id"][index] = img_dict["objects_id"]
                        f["objects_conf"][index] = img_dict["objects_conf"]
                        f["attrs_id"][index] = img_dict["attrs_id"]
                        f["attrs_conf"][index] = img_dict["attrs_conf"]
                        f["num_boxes"][index] = img_dict["num_boxes"]
                        f["boxes"][index] = img_dict["boxes"]
                        f["features"][index] = img_dict["features"]
                        image_to_index[img_dict["img_id"]] = index
                    except Exception as e:
                        json_object = json.dumps(image_to_index, indent=4)
                        with open(index_file, "w") as outfile:
                            outfile.write(json_object)
                        print(e)

        json_object = json.dumps(image_to_index, indent=4)
        with open(index_file, "w") as outfile:
            outfile.write(json_object)


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
    """
    :param net:
    :param im_file: full path to an image
    :param image_id:
    :param conf_thresh:
    :return: all information from detection and attr prediction
    """
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)


    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data


    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    objects_conf = np.max(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)
    attrs_conf = np.max(attr_prob[keep_boxes][:, 1:], axis=1)
    #
    # print("img_id ", type(image_id))
    # print("img_h", type(np.size(im, 0)))
    # print("img_w", type(np.size(im, 1)))
    # print("objects_id", objects.shape)  # int64
    # print("objects_conf", objects_conf.shape)  # float32
    # print("attrs_id", attrs.shape)  # int64
    # print("attrs_conf", attrs_conf.shape)  # float32
    # print("num_boxes", len(keep_boxes))
    # print("boxes", cls_boxes[keep_boxes].shape)  # float32
    # print("features", pool5[keep_boxes].shape) # float32
    #
    # print("img_id ", image_id)
    # print("img_h", np.size(im, 0))
    # print("img_w", np.size(im, 1))
    # print("objects_id", objects)  # int64
    # print("objects_conf", objects_conf)  # float32
    # print("attrs_id", attrs)  # int64
    # print("attrs_conf", attrs_conf)  # float32
    # print("num_boxes", len(keep_boxes))
    # print("boxes", cls_boxes[keep_boxes])  # float32
    # print("features", pool5[keep_boxes])  # float32

    # return {
    #     "img_id": image_id,
    #     "img_h": np.size(im, 0),
    #     "img_w": np.size(im, 1),
    #     "objects_id": base64.b64encode(objects),  # int64
    #     "objects_conf": base64.b64encode(objects_conf),  # float32
    #     "attrs_id": base64.b64encode(attrs),  # int64
    #     "attrs_conf": base64.b64encode(attrs_conf),  # float32
    #     "num_boxes": len(keep_boxes),
    #     "boxes": base64.b64encode(cls_boxes[keep_boxes]),  # float32
    #     "features": base64.b64encode(pool5[keep_boxes])  # float32
    # }

    return {
        "img_id": image_id,
        "img_h": np.size(im, 0),
        "img_w": np.size(im, 1),
        "objects_id": objects,  # int64
        "objects_conf": objects_conf,  # float32
        "attrs_id": attrs,  # int64
        "attrs_conf": attrs_conf,  # float32
        "num_boxes": len(keep_boxes),
        "boxes": cls_boxes[keep_boxes],  # float32
        "features": pool5[keep_boxes]  # float32
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--imgroot', type=str, default='/workspace/images/')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--caffemodel', type=str, default='./resnet101_faster_rcnn_final_iter_320000.caffemodel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()

    args.cfg_file = BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml"  # s = 500
    args.prototxt = BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt"
    h5py_file = "%s_obj36" % args.split
    index_file = "%s_image_to_index.json" % args.split

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    # Load image ids, need modification for new datasets.
    image_ids = load_image_ids(args.imgroot, SPLIT2DIR[args.split])
    # clean_up_download(image_ids)
    # # Generate TSV files, normally do not need to modify

    generate_h5py(args.prototxt, args.caffemodel, image_ids, h5py_file, index_file)

    # args = parse_args()
    #
    # args.cfg_file = BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml"  # s = 500
    # args.prototxt = BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt"
    # args.outfile = "%s_obj36.tsv" % args.split
    #
    # print('Called with args:')
    # print(args)
    #
    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)
    #
    # print('Using config:')
    # pprint.pprint(cfg)
    # assert cfg.TEST.HAS_RPN
    #
    # # Load image ids, need modification for new datasets.
    # image_ids = load_image_ids(args.imgroot, SPLIT2DIR[args.split])
    # # clean_up_download(image_ids)
    # # # Generate TSV files, normally do not need to modify
    #
    # generate_tsv(args.prototxt, args.caffemodel, image_ids, args.outfile)