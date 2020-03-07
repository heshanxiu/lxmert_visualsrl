import h5py
import numpy as np


with h5py.File("/local/shanxiu/lxmert/snap/visualsrl/visualsrl_extract_1/lxmert_feature", "r") as f:
    h5_features = f['features'][0,0]
    print(h5_features)