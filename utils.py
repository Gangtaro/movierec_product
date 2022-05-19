import sys, os, random
import logging
import pandas as pd
import sys
import logging
import torch
from tqdm import tqdm 
from time import time
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
