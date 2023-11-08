import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from glob import glob
from skimage.io import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt

images_train = glob('idd20k_lite/leftImg8bit/train/*/*_image.jpg')
images_val = glob('idd20k_lite/leftImg8bit/val/*/*_image.jpg')
images_test = glob('idd20k_lite/leftImg8bit/test/*/*_image.jpg')
labels_train = [img.replace('leftImg8bit', 'gtFine').replace('_image.jpg', '_label.png') for img in images_train]
labels_val = [img.replace('leftImg8bit', 'gtFine').replace('_image.jpg', '_label.png') for img in images_val]

colors = np.array([[128, 64, 18],      # Drivable
    [244, 35, 232],     # Non Drivable
    [220, 20, 60],      # Living Things
    [0, 0, 230],        # Vehicles
    [220, 190, 40],     # Road Side Objects
    [70, 70, 70],       # Far Objects
    [70, 130, 180],     # Sky
    [0, 0, 0]           # Misc
    ], dtype=int) 

images_train_list = np.zeros((len(images_train), 227, 320, 3), dtype=float)
labels_train_list = np.zeros((len(labels_train), 227, 320), dtype=int)
images_val_list = np.zeros((len(images_val), 227, 320, 3), dtype=float)
labels_val_list = np.zeros((len(labels_val), 227, 320), dtype=int)
images_test_list = np.zeros((len(images_test), 227, 320, 3), dtype=float)

for i, (img, lbl) in enumerate(zip(images_train, labels_train)):
    images_train_list[i] = imread(img)
    labels_train_list[i] = imread(lbl)
    labels_train_list[i][labels_train_list[i] == 255] = 7

for i, (img, lbl) in enumerate(zip(images_val, labels_val)):
    images_val_list[i] = imread(img)
    labels_val_list[i] = imread(lbl)
    labels_val_list[i][labels_val_list[i] == 255] = 7

for i, img in enumerate(images_test):
    images_test_list[i] = imread(img)

images_train_list = images_train_list.transpose((0, 3, 1, 2))
images_val_list = images_val_list.transpose((0, 3, 1, 2))
images_test_list = images_test_list.transpose((0, 3, 1, 2))

images_train_list = torch.from_numpy(images_train_list)
labels_train_list = torch.from_numpy(labels_train_list)
images_val_list = torch.from_numpy(images_val_list)
labels_val_list = torch.from_numpy(labels_val_list)
images_test_list = torch.from_numpy(images_test_list)

# print(images_train_list.shape, labels_train_list.shape)
# print(images_val_list.shape, labels_val_list.shape)

images_train_list = images_train_list / 255.0
images_val_list = images_val_list / 255.0
images_test_list = images_test_list / 255.0

train_dataset = torch.utils.data.TensorDataset(images_train_list, labels_train_list)
val_dataset = torch.utils.data.TensorDataset(images_val_list, labels_val_list)

# Create data loaders for training and validation
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Download pre-trained DeepLabV3 model from torchvision
model = deeplabv3_resnet50(pretrained=True)

# Modify the model for your specific number of classes
num_classes = 8
model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

# Define the device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move the model to the device
model = model.to(device)
model.float()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
loss_list = []

for epoch in range(num_epochs):
    model.train()
    sum_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        #print(images.shape, labels.shape)
        images = images.to(device).float()
        labels = labels.to(device).long()
        
        # Forward pass
        outputs = model(images)['out']
        print(outputs.shape)
        loss = criterion(outputs, labels)
        sum_loss += loss.item() 

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("dsj")

    loss_list.append(sum_loss)   
    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(images)['out']
            _, predicted = torch.max(outputs, 1)
            
            # Calculate metrics
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
    accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {sum_loss}, Val Accuracy: {accuracy}")

plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Save the trained model
torch.save(model.state_dict(), "segmentation_model.pth")

# Load the saved model
model = deeplabv3_resnet50(pretrained=False)
model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load("segmentation_model.pth"))
model = model.to(device)

# Test the model
model.eval()
predicted_list = []

with torch.no_grad():
    for images in images_test_list:
        images = images.unsqueeze(0).to(device)
        outputs = model(images)['out']
        _, predicted = torch.max(outputs, 1)
        predicted_list.append(predicted.cpu().numpy()[0])

# Save the predicted labels as images

for i, predicted in enumerate(predicted_list):
    predicted = colors[predicted]
    predicted = predicted.astype(np.uint8)
    predicted = predicted.transpose((1, 2, 0))
    cv2.imwrite(f"predicted_{i}.png", predicted)

    # View the predicted labels

    plt.imshow(predicted)
    plt.show()