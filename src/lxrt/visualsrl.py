import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from .file_utils import cached_path

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lxmert_vision_layer = nn.Linear()