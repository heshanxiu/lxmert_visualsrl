import h5py
import numpy as np
import argparse
import json
import os
import sys
from tqdm import tqdm


def examine_index_file(original_index, image_to_index):

    if os.path.exists(original_index):
        with open(original_index, "r") as f:
            original = json.load(f)
            original_id = original.keys()
    else:
        raise ReferenceError("Original index file {} not found".format(original_index))
    if os.path.exists(image_to_index):
        with open(image_to_index, "r") as f:
            stored = json.load(f)
            stored_id = stored.keys()
    else:
        raise ReferenceError("Stored index file  {} not found".format(image_to_index))

    # if len(original_id) != len(stored_id):
    #     print("Two index file's lengths are not equal")
    #     print("Length of original id is {}".format(len(original_id)))
    #     print("Length of stored id is {}".format(len(stored_id)))

    missing_list = []
    for i, all_images in enumerate(tqdm(original_id)):
        if all_images not in stored_id:
            # print("missing image in dic {}".format(all_images))
            missing_list.append(all_images)

    # print("In total, we miss {} items from {}".format(len(missing_list), original_index))

    return original, original_id, stored, stored_id, missing_list

def fix_index(h5py_file, original, stored_index):
    original, original_id, stored, stored_id, missing_list = examine_index_file(original, stored_index)
    valid_range = {i: i for i in range(75702)}

    need_fixed = []
    need_fixed.extend(missing_list)
    # print("missing list is {}".format(len(need_fixed)))
    for i, id in enumerate(tqdm(stored_id)):
        if stored[id] > 75702 or stored[id] < 0:
            need_fixed.append(id)
        else:
            if stored[id] in valid_range:
                del valid_range[stored[id]]
            else:
                print("repeated: id")
                need_fixed.append(id)

    # print("{} ids need to be fixed with {} ids left".format(len(need_fixed), len(valid_range)))
    return valid_range, need_fixed



def examine_h5py(h5py_file):
    missing_list = []
    with h5py.File(h5py_file, 'r') as f:
        for i in tqdm(range(len(f['img_h']))):
            if f['img_h'][i] == 0:
                print("Index {} misses info".format(i))
                missing_list.append(i)
        print("Length of h5py file is {}".format(len(f['img_h'])))
    print("In total, we miss {} item from {}".format(len(missing_list), h5py_file))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare a h5py file or an index file to standard')
    parser.add_argument('--h5py', type=str, default='/local/shanxiu/ubert/lxmert/data/visualsrl_imgfeat/train_obj36')
    parser.add_argument('--original_split', type=str, default='/local/shanxiu/ubert/imSitu/train.json')
    parser.add_argument('--index', type=str, default='/local/shanxiu/ubert/lxmert/data/visualsrl_imgfeat/train_image_to_index.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    fix_index(args.h5py, args.original_split, args.index)
