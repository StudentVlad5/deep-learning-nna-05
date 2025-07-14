import os
import random
from collections import defaultdict
from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score

import warnings
# filter warnings
warnings.filterwarnings('ignore')

data_path = 'data/Module_3_Lecture_1_SIGNS_dataset/'
splits = ['train', 'test']

data = [[0, 128, 255], [64, 192, 32], [255, 128, 0]]
plt.imshow(data, cmap='gray')
plt.show()

data = [[[255, 0], [128, 64]], [[128, 255], [64, 0]], [[0, 128], [255, 192]]]
data = np.moveaxis(data, 0, -1)
plt.imshow(data)
plt.show()

data_path = './data/'
splits = ['train', 'test']

# Display images examples

# iterate over train and test folders
for s in splits:
    # list files in the folder with the jpg extension
    files = [f for f in os.listdir(f"{data_path}{s}_signs") if f.endswith('.jpg')]
    print(f'{len(files)} images in {s}')

    # for each image, create a list of the type [class, filename]
    files = [f.split('_', 1) for f in files]

    # group the data by class
    files_by_sign = defaultdict(list)
    for k, v in files:
        files_by_sign[k].append(v)

    # take random 4 images of each class
    for k, v in sorted(files_by_sign.items()):
        print(f'Number of examples for class {k}:', len(v))

        # display several examples of images from the training sample   
        if s == 'train':        
            random.seed(42)

            imgs_path = random.sample(v, 4)
            imgs_path = [os.path.join(data_path, f'{s}_signs/{k}_{p}') for p in imgs_path]

            # read the image using the opencv library
            imgs = [cv2.imread(p) for p in imgs_path]
            # matplotlib expects img in RGB format but OpenCV provides it in BGR       
            # transform the BGR image into RGB
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

            # create a figure for display
            fig = plt.figure(figsize=(7, 2))
            grid = ImageGrid(
                fig, 111, 
                nrows_ncols=(1, 4)
            )
            # display the image
            for ax, img in zip(grid, imgs):
                ax.imshow(img)

            fig.suptitle(f'Class {k}, {s.capitalize()} split')
            plt.show()  # <-- викликаємо plt.show() після кожного класу


class SIGNSDataset(Dataset):
    def __init__(self, data_dir, transform):      
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        # self.labels = [int(filename.split('/')[-1][0]) for filename in self.filenames]
        self.labels = [int(os.path.basename(f)[0]) for f in self.filenames]
        self.transform = transform
# Мета: завантажити і зберегти імена файлів .jpg, а також витягнути мітки (label) з першого символу назви файлу (наприклад, 2_image1.jpg → label 2).

# Проблема: на Windows або кросплатформено краще використовувати os.path.basename(filename) замість filename.split('/').

    def __len__(self):
        # returns the size of the dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        # open the image, apply transformations and
        # return an image with a class label
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

# Трансформації зображень

train_transformer = T.Compose([
    T.Resize(64),              # зменшення зображення до 64x64
    T.RandomHorizontalFlip(),  # випадкове горизонтальне віддзеркалення
    T.ToTensor()])             # перетворення в тензор PyTorch

eval_transformer = T.Compose([
    T.Resize(64),
    T.ToTensor()])

# 3. Dataloader-и

train_dataset = SIGNSDataset(f'{data_path}train_signs/', train_transformer)
test_dataset = SIGNSDataset(f'{data_path}test_signs/', eval_transformer)

# 3 згорткові шари → зменшення розміру (64 → 32 → 16 → 8).

# Після згорток: flatten і два повнозв’язні (dense) шари.

# Останній шар → 6 класів → log_softmax (хоча краще прибрати log_softmax, якщо ви використовуєте CrossEntropyLoss, бо вона вже включає softmax).

class BaselineModel(nn.Module):

  def __init__(self, ): 
    super().__init__()
    self.num_channels = 32
     
    # convolution base
    self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)

    # classification layers
    self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
    self.fc2 = nn.Linear(self.num_channels*4, 6)    

  def forward(self, s):
    #                         -> batch_size x 3 x 64 x 64
    s = self.conv1(s)                  # batch_size x num_channels x 64 x 64
    s = F.relu(F.max_pool2d(s, 2))           # batch_size x num_channels x 32 x 32
    s = self.conv2(s)                  # batch_size x num_channels*2 x 32 x 32
    s = F.relu(F.max_pool2d(s, 2))           # batch_size x num_channels*2 x 16 x 16
    s = self.conv3(s)                  # batch_size x num_channels*4 x 16 x 16
    s = F.relu(F.max_pool2d(s, 2))           # batch_size x num_channels*4 x 8 x 8

    # flatten the output for each image
    s = s.view(-1, 8*8*self.num_channels*4)       # batch_size x 8*8*num_channels*4

    # apply 2 fully connected layers with dropout
    s = F.relu(self.fc1(s))               # batch_size x self.num_channels*4
    s = self.fc2(s)                   # batch_size x 6

    return s  # або F.log_softmax(s, dim=1) тільки якщо ви точно знаєте, що loss буде NLLLoss


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)  # ✅ правильний виклик


train_dataloader = DataLoader(train_dataset,
                              batch_size=32, 
                              shuffle=True,
                              num_workers=4)

test_dataloader = DataLoader(test_dataset,
                             batch_size=32, 
                             shuffle=False,
                             num_workers=1)

model = BaselineModel().to(device)

model

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define loss function
criterion = nn.CrossEntropyLoss().to(device)

train_losses = []
train_accs = []

test_losses = []
test_accs = []

num_epochs = 15

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_epoch_loss = []
    train_epoch_acc = []

    for i, (train_batch, labels_batch) in tqdm(enumerate(train_dataloader)):
        if cuda:
            train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)

        output_batch = model(train_batch)
        loss = criterion(output_batch, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.cpu().detach().numpy()
        batch_acc = balanced_accuracy_score(
            np.argmax(output_batch.cpu().detach().numpy(), axis=1),
            labels_batch.cpu().detach().numpy()
        )

        train_epoch_loss.append(batch_loss)
        train_epoch_acc.append(batch_acc)

    print(f'Train epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(train_epoch_loss):.4f}, Acc: {np.mean(train_epoch_acc):.4f}')
    train_losses.append(np.mean(train_epoch_loss))
    train_accs.append(np.mean(train_epoch_acc))

    # ======================
    #       Evaluation
    # ======================
    model.eval()
    test_epoch_loss = []
    test_epoch_acc = []

    with torch.no_grad():
        for i, (test_batch, labels_batch) in enumerate(test_dataloader):
            if cuda:
                test_batch, labels_batch = test_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)

            output_batch = model(test_batch)
            loss = criterion(output_batch, labels_batch)

            batch_loss = loss.cpu().detach().numpy()
            batch_acc = balanced_accuracy_score(
                np.argmax(output_batch.cpu().detach().numpy(), axis=1),
                labels_batch.cpu().detach().numpy()
            )

            test_epoch_loss.append(batch_loss)
            test_epoch_acc.append(batch_acc)

    print(f'Test epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(test_epoch_loss):.4f}, Acc: {np.mean(test_epoch_acc):.4f}')
    test_losses.append(np.mean(test_epoch_loss))
    test_accs.append(np.mean(test_epoch_acc))


# compute model output and loss
output_batch = model(train_batch)

loss = criterion(output_batch, labels_batch)

# clear previous gradients, compute gradients of all variables wrt loss
optimizer.zero_grad()
loss.backward()

# performs updates using calculated gradients
optimizer.step()
    
batch_loss = loss.cpu().detach().numpy()
batch_acc = balanced_accuracy_score(np.argmax(output_batch.cpu().detach().numpy(), axis=1),
                    labels_batch.cpu().detach().numpy())
    
train_epoch_loss.append(batch_loss)
train_epoch_acc.append(batch_acc)
     
   
print(f'Train epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(train_epoch_loss):.4f}, Acc: {np.mean(train_epoch_acc)}')
train_accs.append(np.mean(train_epoch_acc))
train_losses.append(np.mean(train_epoch_loss))

# Eval step

model.eval()

test_epoch_loss = []
test_epoch_acc = []

with torch.no_grad():
    
    for i, (test_batch, labels_batch) in enumerate(test_dataloader):
        if cuda:
            test_batch, labels_batch = test_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)

        # compute model output and loss
        output_batch = model(test_batch)

        loss = criterion(output_batch, labels_batch)

        batch_loss = loss.cpu().detach().numpy()
        batch_acc = balanced_accuracy_score(np.argmax(output_batch.cpu().detach().numpy(), axis=1),
                        labels_batch.cpu().detach().numpy())

        test_epoch_loss.append(batch_loss)
        test_epoch_acc.append(batch_acc)
print(f'Test epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(test_epoch_loss):.4f}, Acc: {np.mean(test_epoch_acc)}')     
test_accs.append(np.mean(test_epoch_acc))
test_losses.append(np.mean(test_epoch_loss))

plt.figure(figsize=(4, 3))
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Metric')
plt.show()


plt.figure(figsize=(4, 3))
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.show()

