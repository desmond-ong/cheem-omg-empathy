## Script for extracting VGG face features from our subjects and actors in the videos ##
## Download the pretrained VGG-Face pytorch model from http://www.robots.ox.ac.uk/~albanie/pytorch-models.html ##
## Place them in the dir ./pretrained_models/ ##


from __future__ import print_function, division
import sys
sys.path.insert(0,'./feature_extraction/')
sys.path.insert(0,'./utils/')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import time
import copy
import os
import sys
import argparse
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
import pickle
from models import VGG_Face_torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_gpu = torch.cuda.is_available()

base_dir = './data/Testing'   ##change the base directory for train, val and test

mean = [0.4856586910840433, 0.4856586910840433, 0.4856586910840433]
std = [0.14210993338737993, 0.14210993338737993, 0.14210993338737993]

class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()
        self.pre_model = nn.Sequential(*list(model.children())[:-1])
       
    def forward(self, x):
        x = self.pre_model(x)
        return x

print('Model setup')

print("Loading checkpoint model for feature extraction...")

ckpt_path = './pretrained_models/VGG_Face_torch.pth'
model_emotion = VGG_Face_torch
model_emotion.load_state_dict(torch.load(ckpt_path))
model = VGG_Net(model_emotion).cuda()

print("Consisting a feature extractor from the model...")
extractor=model
extractor.eval()

if use_gpu:
    model.cuda()
    extractor.cuda()
    cudnn.benchmark = True

print("Feature Extraction")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

file_path = os.path.join(base_dir,'Extracted_Faces')
img_dirs = os.listdir(file_path)
vgg_feat_dir = os.path.join(base_dir, 'Visual_Features/Subject')    ### Change Subject folder to Actor if you want to extract Actor features
if not os.path.isdir(vgg_feat_dir):
	os.mkdirs(vgg_feat_dir)

def main():
	for i in range((len(img_dirs)):
	    image_files = os.path.join(file_path, img_dirs[i], 'Subject')
	    image_frames = os.listdir(image_files)
	    feat_dirs = os.path.join(vgg_feat_dir, img_dirs[i])
	    os.mkdir(feat_dirs)
	    for j in range(len(image_frames)):
		image_path = os.path.join(image_files, image_frames[j])
		feat_name = image_frames[j]
		feat_name = feat_name[0:len(feat_name)-4]
		image = Image.open(image_path)
		# transform input image to tensor
		if test_transform is not None:
		    image = test_transform(image)
		inputs = image
		inputs = Variable(inputs, volatile=True)

		if use_gpu:
		    inputs = inputs.cuda()
		inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front

		# extract VGG face features from the fc7 layer, 4096-d feature vectors 
		output = extractor(inputs)
		feat = output.cpu()
		feat = feat.detach().numpy()
		feats = os.path.join(feat_dirs,(str(feat_name)+'.npy'))

		# save visual raw features
		np.save(feats, feat)
		print('Images processed,', j)

if __name__ == '__main__':
    main()


