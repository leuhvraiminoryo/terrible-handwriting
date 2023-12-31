import torch.nn as nn
import torch.nn.functional as F
import scripts.dataset as ds
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 26),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

classes = [chr(ord("a")+i) for i in range(26)]
PATH = 'aux/letters_net.pth'

transform = transforms.Compose(
        [transforms.Normalize((0.5,), (0.5,))])

def train_net():
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    c = ds.CustomImageDataset("data/train.csv", "data/train_pics", transform=transform)
    bs=8
    trainloader = torch.utils.data.DataLoader(c, batch_size=bs, shuffle=True, num_workers=2)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)


def get_network(retrain=False):
    if retrain:
        train_net()

    net = Net()
    net.load_state_dict(torch.load(PATH))

    return net

def test(net):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    c = ds.CustomImageDataset("data/test.csv", "data/test_pics", transform=transform)
    bs=8
    testloader = torch.utils.data.DataLoader(c, batch_size=bs, shuffle=True, num_workers=2)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total = sum(total_pred.values())
    correct = sum(correct_pred.values())
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class:  {classname:2s} is {accuracy:.1f} %')

def test_single_image(network, path="aux/to_predict.png"):
    transform = transforms.Compose([transforms.PILToTensor()])

    im = Image.open(path)
    image_tensor = transform(im)

    im_ten = image_tensor.type(torch.float)

    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    im_ten = transform(im_ten)

    out = network.forward(im_ten)
    _, ind = out.max(1)

    print(f"predicted for {path} : {classes[ind]}")

