
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 08:54:28 2021

@author: Moona Mazher
"""
       
#%% dataloader class
import os
import torch
from torch.utils.data import Dataset

##########################################################new dataset loader ######################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
#import keras
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import torch
from torch.utils.data import Dataset
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
## please refere to give for dataset prepartion
#https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

class Dataset(BaseDataset):
    
    #CLASSES = ['leftlung', 'rightlung', 'disease','unlabelled']
    #CLASSES = ['leftlung', 'rightlung','unlabelled']
    CLASSES = ['externalcerebrospinalfluid', 
               'greymatter',
               'whitematter',
               'ventricles',
               'cerebellum',
               'deepgreymatter',
               'brainstem',
               'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            transform=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.transform=transform
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #image = cv2.imread(self.images_fps[i])
        image = Image.open(self.images_fps[i])
        image=(np.asarray(image) / 65536).astype('float32')
        #image=torch.FloatTensor(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image=cv2.convertScaleAbs(image)
        image_name = self.ids[i]
        mask = cv2.imread(self.masks_fps[i], 0)
        #mask = io.imread(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            mask=mask.transpose(2,0,1) # n_class*w*H
            #mask=torch.uint8(mask)
            #mask=mask.transpose(2,0,1) # n_class*w*H
        #onehot_label=torch.FloatTensor(img_label_onehot)
        #print(onehot_label.shape)
        #mask=torch.uint8(mask)
        mask=torch.from_numpy(mask).type(torch.uint8)
            
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.transform:
            image=self.transform(image)
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


   
imagepath='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\trainingdata\\images'
maskpath='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\trainingdata\\masks'    
x_train_dir=imagepath
y_train_dir=maskpath

imagepath_valid='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validationdata\\images'
maskpath_valid='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validationdata\\masks'    
x_valid_dir=imagepath_valid
y_valid_dir=maskpath_valid


#data=data[:,:,0]  
# Lets look at data we have
from torchvision import transforms
# transform=transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize(mean=[0.485,0.456,0.406],
#                                                   std=[0.229,0.224,0.225])])

transform=transforms.Compose([transforms.ToTensor(),])

from torch.utils.data import DataLoader
dataset_train = Dataset(x_train_dir, y_train_dir, classes=['externalcerebrospinalfluid', 'greymatter','whitematter','ventricles','cerebellum','deepgreymatter','brainstem'],transform=transform)
dataset_valid = Dataset(x_valid_dir, y_valid_dir, classes=['externalcerebrospinalfluid', 'greymatter','whitematter','ventricles','cerebellum','deepgreymatter','brainstem'],transform=transform)
   
len(dataset_train) 
imges,masks=dataset_train[0]
import random
ix = random.randint(0, len(dataset_train))
img, mask= dataset_train[ix]
fig, ax = plt.subplots(dpi=50)
ax.imshow(img[0], cmap="gray")
ax.axis('off')
mask = torch.argmax(mask, axis=0).float().numpy()
mask[mask == 0] = np.nan
ax.imshow(mask, alpha=0.5)
plt.show()

imges.shape, imges.dtype, imges.max(), imges.min()
imges1=imges[0,:,:].numpy()
img, mask= dataset_train[ix]
#fig, ax = plt.subplots(dpi=50)
plt.imshow(img[0], cmap="gray")

#mask = torch.argmax(mask, axis=0).float().numpy()
# mask[mask == 0] = np.nan
# plt.imshow(mask)


#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
# valid_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)

data={'train':dataset_train,
      'val':dataset_valid}
## check dataset image shape and mask
imgs, masks = next(iter(data['train']))
imgs.shape, masks.shape
#################### take the batch size and prepare dataloader ######
batch_size=25
dataloader = {
    'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, pin_memory=True),
}
imgs, masks = next(iter(dataloader['train']))
imgs.shape, masks.shape
# imgs.dtype
# imgs= torch.squeeze(imgs, 1)
# import matplotlib.pyplot as plt

# r, c = 5, 5
# fig = plt.figure(figsize=(5*r, 5*c))
# for i in range(r):
#     for j in range(c):
#         ix = c*i + j
#         ax = plt.subplot(r, c, ix + 1)
#         ax.imshow(imgs[ix].squeeze(0), cmap="gray")
#         mask = torch.argmax(masks[ix], axis=0).float().numpy()
#         mask[mask == 0] = np.nan
#         ax.imshow(mask, alpha=0.5)
#         ax.axis('off')
# plt.tight_layout()
# plt.show()
############################################### DensUent model #######################################
#%% densUnet model
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class up_in(nn.Sequential):
    def __init__(self, num_input_features1, num_input_features2, num_output_features):
        super(up_in, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv1_1', nn.Conv2d(num_input_features1, num_input_features2,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('conv3_3', nn.Conv2d(num_input_features2, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        x = self.conv1_1(x)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class upblock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class up_out(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(up_out, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout2d(p=0.3)
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, y):
        y = self.up(y)
        y = self.conv3_3(y)
        y = self.dropout(y)
        y = self.norm(y)
        y = self.relu(y)
        return y


class DenseUNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, num_channels, num_classes,growth_rate=48, block_config=(6, 12, 36, 24),
                num_init_features=96, bn_size=4, drop_rate=0,):

        super(DenseUNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.up1 = up_in(48*44, 48*46, 48*16)
        self.up2 = upblock(48*16, 48*8)
        self.up3 = upblock(48*8, 96)
        self.up4 = upblock(96,96)
        self.up5 = up_out(96,64)
        self.outconv = outconv(64,num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features.conv0(x)
        x0 = self.features.norm0(features)
        x0 = self.features.relu0(x0)
        x1 = self.features.pool0(x0)
        x1 = self.features.denseblock1(x1)
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        x4 = self.features.transition3(x3)
        x4 = self.features.denseblock4(x4)
        
        y4 = self.up1(x3, x4)
        y3 = self.up2(x2, y4)
        y2 = self.up3(x1, y3)
        
        y1 = self.up4(x0, y2)
        y0 = self.up5(y1)
        out = self.outconv(y0)
        # out = F.softmax(out, dim=1)
        
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
model=DenseUNet(3, 8)
#print(modelUdense)



########### define the training and testing function ###########
import os
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
lr = 3e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr=3e-4
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

batch_size=24
#%% define training and validation function
#second training function for optimizing the model
import pandas as pd
def IoU(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    gt = gt > th
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    ious = (intersection + eps) / union
    return torch.mean(ious).item()


# def iou(outputs, labels):
#     # check output mask and labels
#     outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
#     SMOOTH = 1e-6
#     # BATCH x num_classes x H x W
#     B, N, H, W = outputs.shape
#     ious = []
#     for i in range(N-1): # we skip the background
#         _out, _labs = outputs[:,i,:,:], labels[:,i,:,:]
#         intersection = (_out & _labs).float().sum((1, 2))  
#         union = (_out | _labs).float().sum((1, 2))         
#         iou = (intersection + SMOOTH) / (union + SMOOTH)  
#         ious.append(iou.mean().item())
#     return np.mean(ious)

from tqdm import tqdm
from collections import OrderedDict

##################### training function ##########
def train(dataloader, model, criterion, optimizer, epoch, scheduler=None):
    bar = tqdm(dataloader['train'])
    losses_avg, ious_avg = [], []
    train_loss, train_iou = [], []
    model.to(device)
    model.train()
    for imgs, masks in bar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        y_hat = model(imgs)
        loss = criterion(y_hat, masks)
        loss.backward()
        optimizer.step()
        ious = IoU(y_hat, masks)
        train_loss.append(loss.item())
        train_iou.append(ious)
        #bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
    losses_avg=np.mean(train_loss)
    ious_avg=np.mean(train_iou)
    
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    return log

def validate(dataloader, model, criterion):
    bar = tqdm(dataloader['val'])
    test_loss, test_iou = [], []
    losses_avg, ious_avg = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs, masks in bar:
            imgs, masks = imgs.to(device), masks.to(device)
            y_hat = model(imgs)
            loss = criterion(y_hat, masks)
            ious = IoU(y_hat, masks)
            test_loss.append(loss.item())
            test_iou.append(ious)
            bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    losses_avg=np.mean(test_loss)
    ious_avg=np.mean(test_iou)
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    
    return log
criterion = torch.nn.BCEWithLogitsLoss()
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])
early_stop=20
epochs=10000
best_iou = 0
name='DensUnet'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_log = train(dataloader, model, criterion, optimizer, epoch)
    #train_log = train(train_loader, model, optimizer, epoch)
    # evaluate on validation set
    #val_log = validate(val_loader, model)
    val_log =validate(dataloader, model, criterion)
    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'%(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    tmp = pd.Series([epoch,lr,train_log['loss'],train_log['iou'],val_log['loss'],val_log['iou']], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %name, index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), 'models/%s/model.pth' %name)
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache()
print("done training")

               