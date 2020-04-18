# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from tasks.visualsrl_model import VisualSRLModel
from tasks.visualsrl_data import VisualSRLTorchDataset

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

SPLIT2NUM = {
    'train': 75702,
    'dev': 25200,
    'test': 25200,
}

def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:

    #need this
    tset = VisualSRLTorchDataset(splits)
    evaluator = None
    #need this
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(None, loader=data_loader, evaluator=evaluator)


class VisualSRL:
    def __init__(self):
        # Data sets modified get visual srl data loader & features here

        self.train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)
        self.model = VisualSRLModel()

        # Load pre-trained weights
        # which pre-trained to use?
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def extract_feature(self, train_tuple: DataTuple):
        # get every feature from visualsrl
        #input into lxmert
        #call lxmert to let it run, add functionality in visual-encoder to record each layer's representation

        self.model.eval()
        _, loader, _ = train_tuple

        h5_file_path = os.path.join(args.output, "lxmert_feature")
        if not os.path.exists(h5_file_path):
            with h5py.File(h5_file_path, 'a') as h5_file:
                num_examples = SPLIT2NUM[args.train]
                h5_features = h5_file.create_dataset('features', (num_examples, 5, 36, 768), dtype=np.float32)

        with h5py.File(h5_file_path, 'a') as h5_file:
            image_index_dic = {}
            for i, datum_tuple in enumerate(tqdm(loader)):
                img_name, feats, boxes = datum_tuple  # Avoid seeing ground truth
                try:
                    with torch.no_grad():
                        feats, boxes = feats.cuda(), boxes.cuda()
                        ((lang_feats, visn_feats), _) = self.model(feats, boxes)
                        image_index_dic[str(img_name[0])] = i
                        if lang_feats is None:
                            for j, each_layer_feat in enumerate(visn_feats):
                                h5_file['features'][i, j] = each_layer_feat.cpu().numpy()
                except Exception as e:
                    json_object = json.dumps(image_index_dic, indent=4)
                    json_file_path = os.path.join(args.output, "image_id_to_index.json")
                    with open(json_file_path, "w") as outfile:
                        print("json file path {}".format(json_file_path))
                        outfile.write(json_object)
                    print(e)

        json_object = json.dumps(image_index_dic, indent=4)
        json_file_path = os.path.join(args.output, "image_id_to_index.json")
        with open(json_file_path, "w") as outfile:
            print("json file path {}".format(json_file_path))
            outfile.write(json_object)


    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    visualsrl = VisualSRL()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        visualsrl.load(args.load)

    visualsrl.extract_feature(visualsrl.train_tuple)

