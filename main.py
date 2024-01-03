import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import scripts.decomp as dec
import scripts.network as net


'''
traite les images en 'alphabet_strips/' pour remplir/remplacer les éléments du dossier 'data/'
à ne décommenter que si nécessaire
'''
#dec.treat_data("alphabet_strips/")

nn = net.get_network()
net.test_single_image(nn)