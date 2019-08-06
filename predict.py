import argparse
from model import load_checkpoint, predict

# Setup argparse arguments
parser = argparse.ArgumentParser()

parser.add_argument('img_path',type=str)
parser.add_argument('check_point',type=str)
parser.add_argument('--topk',type=int,default=3)
parser.add_argument('--category_name',type=str,default='./cat_to_name.json')
parser.add_argument('--gpu',action='store_true')

arg = parser.parse_args()

print('Predicting...') 
model = load_checkpoint(arg.check_point) 
probs, classes = predict(arg.img_path, model, arg.topk, arg.gpu,arg.category_name)
output = dict(zip(classes,probs))
print(output)
