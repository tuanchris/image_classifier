# Import libraries
import argparse
from model import create_model, train_model, save_checkpoint
from utils import load_data

# Setup argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument('data_dir',type=str)
parser.add_argument('--save_dir',type=str,default = './')
parser.add_argument('--arch',type=str, default = 'vgg16')
parser.add_argument('--lr',type=float, default = '0.01')
parser.add_argument('--hidden_units',type=int, default = 512)
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--gpu',action='store_true')

arg = parser.parse_args()

if __name__ == '__main__':
    print('Loading data')
    trainloader, validloader, testloader, class_to_idx = load_data(arg.data_dir)

    print('Creating model')
    model = create_model(arg.arch,arg.hidden_units)

    print('Training model')
    train_model(model,arg.epochs, arg.lr,arg.gpu, trainloader,validloader)

    print('Saving model')
    model.class_to_idx = class_to_idx
    save_checkpoint(model,arg.save_dir+arg.arch+'.pth', arg.arch, arg.hidden_units,class_to_idx)
