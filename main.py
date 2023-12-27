import scripts.dataset as ds

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

'''
c = ds.CustomImageDataset("data/shite.csv", "data/train_pics")

labels_map = {0 : "a", 1 : "b", 2 : "c"}

figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(c), size=(1,)).item()
    img, label = c[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

training_dataloader = DataLoader(c, batch_size=2, shuffle=True)
'''

import scripts.decomp as dec 
dec.load_all_strips("alphabet_strips/")