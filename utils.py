import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from torch import autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import sys, os, math, pickle, warnings, random
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
from pprint import pprint
import learn2learn as l2l
from tqdm import tqdm, trange
from time import time

torch.autograd.set_detect_anomaly(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

def init_weights(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module

def seed(np_seed=11041987, torch_seed=20051987):
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def cuda_memory(dev):
    t = torch.cuda.get_device_properties(dev).total_memory
    c = torch.cuda.memory_cached(dev)
    a = torch.cuda.memory_allocated(dev)
    return t, c, a, t - c - a

def count_network_params(net):
    numel = 0
    for p in net.parameters():
        numel += torch.numel(p)
    return numel