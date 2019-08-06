import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import numpy as np
import torchvision.models as models
import json
from utils import process_image

def create_model(model='vgg16',hidden_units = 512, output_size = 102):
    pytorch_models = {'resnet18' : 'models.resnet18()',
        'alexnet' : 'models.alexnet()',
        'vgg16' :' models.vgg16()',
        'squeezenet' : 'models.squeezenet1_0()',
        'densenet' : 'models.densenet161()',
        'inception' : 'models.inception_v3()',
        'googlenet' : 'models.googlenet()',
        'shufflenet' : 'models.shufflenet_v2_x1_0()',
        'mobilenet' : 'models.mobilenet_v2()',
        'resnext50_32x4d' : 'models.resnext50_32x4d()'}
    
    try:
        model = eval(pytorch_models[model])
    except:
        raise ValueError('No pretrained model ' + model + ' found!')

    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,hidden_units)),
        ('relu1',nn.ReLU()),
        ('fc2',nn.Linear(hidden_units,output_size)),
        ('output',nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

def train_model(model,epochs,lr,gpu,trainloader,validloader):
    device = torch.device("cuda" if gpu else 'cpu')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = lr)

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

def save_checkpoint(model,output_dir, arch, hidden_units, class_to_idx):
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'class_to_idx' :class_to_idx,
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint, output_dir)
    
def load_checkpoint(path): 
    cp = torch.load(path) 
    model = create_model(cp['arch'],cp['hidden_units']) 
    model.class_to_idx = cp['class_to_idx']
    
    return model

def predict(image_path, model, topk,gpu,json_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device("cuda" if gpu else 'cpu')
    model.to(device);
    
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    output = torch.exp(model(img_tensor)).topk(topk)
    probs, classes = output
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    classes = np.vectorize(idx_to_class.get)(classes)
    classes = np.vectorize(cat_to_name.get)(classes)[0]
    probs = probs.cpu().detach().numpy()[0]
    
    return probs,classes