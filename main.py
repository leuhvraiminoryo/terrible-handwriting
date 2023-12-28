import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import scripts.decomp as dec 
dec.load_all_strips("alphabet_strips/")

