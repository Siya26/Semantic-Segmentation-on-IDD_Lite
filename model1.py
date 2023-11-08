import numpy as np
from imageio import imread, imsave
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import accuracy_score

images_train = glob('idd20k_lite/leftImg8bit/train/*/*_image.jpg')
labels_list = [img.replace('leftImg8bit', 'gtFine').replace('_image.jpg', '_label.png') for img in images_train]

colors = np.array([[128, 64, 18],      # Drivable
    [244, 35, 232],     # Non Drivable
    [220, 20, 60],      # Living Things
    [0, 0, 230],        # Vehicles
    [220, 190, 40],     # Road Side Objects
    [70, 70, 70],       # Far Objects
    [70, 130, 180],     # Sky
    [0, 0, 0]           # Misc
    ], dtype=int)          

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 4, 5, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            
            nn.Conv2d(4, 8, 5, 2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            # nn.Linear(32 * 3 * 6, 8),
            # nn.ReLU(True),
            # nn.Linear(128, 32),
            # nn.ReLU(True),
            # nn.Linear(32, 8),
            # nn.ReLU(True),
        )

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 5, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.dec1 = nn.Sequential(
            # nn.Linear(8, 32 * 3 * 6),
            # nn.ReLU(True),
            # nn.Linear(32, 128),
            # nn.ReLU(True),
            # nn.Linear(128, 32 * 3 * 6),
            # nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 4, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 5, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #nn.MaxUnpool2d(2, 2),            
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 7, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 8, 9, 2, bias=False),
            #nn.Softmax(dim=1),    
        )

        self.indices = []
        
    def forward(self, x):
        #print(x.shape)
        x = self.enc1(x)
        #print(x.shape)
        x, indices = self.pool(x)
        #print(x.shape)
        # self.indices.append(indices)
        x = self.enc2(x)

        #print(x.shape)
        x = self.dec1(x)
        x = x[:, :, 1:, :]
        #print(x.shape)
        # indices2 = self.indices.pop()  # Get indices from the stack
        
        x = self.unpool(x, indices)
        #print(x.shape)
        x = self.dec2(x)
        #print(x.shape)
        x = x[:, :, 1:, 3:230]
        return x


epochs = 10

images = np.zeros((len(images_train), 227, 320, 3), dtype=float)
labels = np.zeros((len(labels_list), 227, 320), dtype=int)

for i in range(len(images_train)):
    images[i] = imread(images_train[i])
    labels[i] = imread(labels_list[i])
    labels[i][labels[i] == 255] = 7

images = torch.tensor(images, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)

model =  Net()
batch_size = 256

def view(image, label):
    color_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=int)
    for i in range(7):
        color_image[label == i] = colors[i]

    color_image[label == 255] = colors[7]
    plt.imshow(image)
    plt.imshow(color_image, alpha=0.8)
    plt.show()

def IoU(pred, label):
    label = np.transpose(label, (1, 2, 0))
    intersection = np.logical_and(pred, label)
    union = np.logical_or(pred, label)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def train():
    loss_list = []
    acc_list = []
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i in range(len(images) // batch_size):
            # get the inputs; data is a list of [inputs, labels]
            input, label = images[i*batch_size:(i+1)*batch_size], labels[i*batch_size: (i+1)*batch_size]
            input = input.transpose(1, 3)
            # input = input.unsqueeze(0)
            # label = label.unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(input)

            # if epoch == 9:
            #     view(input[i].transpose(0, 2).detach().int().numpy(), out)
            label = label.transpose(2, 1)
            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

        scheduler.step()
        #get accuracy
        true_label = labels.transpose(0, 1).detach().int().numpy()
        output = model(images.transpose(1, 3))
        out = output.detach().numpy().argmax(axis=1)
        accuracy = accuracy_score(true_label.flatten(), out.flatten())
        acc_list.append(accuracy)

        loss = criterion(output, labels.transpose(2, 1))
        #get iou
        iou = IoU(out, true_label)

        print("Epoch: ", epoch, "Loss: ", loss.item(), "Accuracy: ", accuracy, "IoU: ", iou)
        loss_list.append(loss.item())

    #torch.save(model.state_dict(), 'model.pth')
    return loss_list, acc_list

def test():
    images_test = glob('idd20k_lite/leftImg8bit/test/*/*_image.jpg')
    images_list = np.zeros((len(images_test), 227, 320, 3), dtype=float)
    predicted = np.zeros((len(images_test), 227, 320), dtype=int)

    for i in range(len(images_test)):
        images_list[i] = imread(images_test[i])

    images_list = torch.tensor(images_list, dtype=torch.float)
    #print(images.shape)
    images_list = images_list.transpose(1, 3)
    #print(images.shape)
      
    #model.load_state_dict(torch.load('model.pth'))  
    model.eval()
    #print('Model loaded')
    for i in range(len(images_list)):
        #print(images_list[i])
        img = torch.unsqueeze(images_list[i], dim=0)
        output = model(img)[0].transpose(1, 2).detach().numpy().argmax(axis=0)

        view(images_list[i].transpose(0, 2).detach().int().numpy(), output)
        #get accuracy
        true_label = images_list[i].argmax(dim=0).transpose(0, 1).detach().int().numpy()
        accuracy = accuracy_score(true_label.flatten(), output.flatten())
        print("Accuracy: ", accuracy)

        pred_path = images_test[i].replace('idd20k_lite', 'preds').replace('leftImg8bit/test/', '').replace('_image.jpg', '_label.png')

        os.makedirs(os.path.dirname(os.path.relpath(pred_path)), exist_ok=True)
        img = Image.fromarray(output.astype(np.uint8))      
        img.save(pred_path)

loss, accuracy = train()
plt.plot(loss)
plt.show()

plt.plot(accuracy)
plt.show()
test()