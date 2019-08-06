
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import vgg16
from PIL import Image
import numpy as np

def load_data(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load and transform training datasets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                          ])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                          ])

    train_data = datasets.ImageFolder(train_dir,train_transforms)
    test_data = datasets.ImageFolder(test_dir,test_transforms)
    valid_data = datasets.ImageFolder(valid_dir,valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)

    class_to_idx = train_data.class_to_idx

    return trainloader, validloader, testloader, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    torch.set_default_tensor_type('torch.FloatTensor')
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    resize = (256,256*(img.size[1])/img.size[0]) if img.size[0]<img.size[1] else (256*(img.size[0])/img.size[1],256)
    img.thumbnail(resize)

    w, h = img.size
    new_w = new_h = 224
    left = (w-new_w)/2
    right = (w+new_w)/2
    top = (h-new_h)/2
    bot = (h+new_h)/2
    img = img.crop((left,top,right,bot))

    np_image = np.array(img)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image).float()
