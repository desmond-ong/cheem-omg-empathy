import torch
import os
import sys
import torchvision
from torch import nn
import torchvision.transforms as transforms
sys.path.insert(0,'./feature_extraction/')
sys.path.insert(0,'./utils/')
import numpy as np
from models import VGG_Face_torch
import argparse
import torch.optim as optim
from torch.autograd import Variable
import data_loader
from data_loader import OMGEmpathyDataset
import torch.nn.functional as F
from calculateCCC import calculateCCC, mse, f1
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import csv
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from train import VGG_LSTM
import re
import pandas as pd

base_dir = './data/Testing' #Training'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print("#GPUs: {}".format(torch.cuda.device_count()))
    print("Current GPU: {}".format(torch.cuda.current_device()))

use_gpu = torch.cuda.is_available()

print("Loading checkpoint model...")
lstm_input_size = 4096
h1 = 512
output_dim=1
num_layers =1
batchsize=1 
seq_len = 60 ## best results for seq length of 60
kwargs = {'num_workers': 4, 'pin_memory': True}

model = VGG_LSTM(lstm_input_size, h1, batch_size=batchsize, output_dim=output_dim, num_layers=num_layers,use_cuda=True,feature=False)

total_seq_pred = []
pred_feat_embed = os.path.join(base_dir,'Visual_Embeddings/Subject/')
pred_output_file = os.path.join(base_dir, 'Predictions_Visual')
if not os.path.isdir(pred_output_file):
    os.mkdir(pred_output_file)
    os.mkdir(pred_feat_embed)

ckpt_path = './models/'+ 'model_1100.pt'

test_data_dir = os.path.join(base_dir, 'Avg_Visual_Features/Subject')
test_csv_dir = os.path.join('./data/Validation', 'Annotations')   ### dummy annotations for Test dataset

transformed_dataset = OMGEmpathyDataset(csv_dir=test_csv_dir,root_dir=test_data_dir, batch=batchsize)
test_dataloader = DataLoader(transformed_dataset, batch_size=batchsize*seq_len, shuffle=False,**kwargs)

model.load_state_dict(torch.load(ckpt_path))
model.to(device)

for batch_idx, (data,target) in enumerate(test_dataloader):
 
    data, target = Variable(data).cuda(), Variable(target).cuda()

    data = data.view(batchsize, data.size(0),-1)
    
    ## Extract feature embeddings from the VGG_LSTM trained model, with feature=True
    """
    feats = model(data)
    feats = feats.cpu()
    feats = feats.detach().numpy()
    feats = np.squeeze(feats,0)
    """
    mean_pred = model(data)
    mean_pred = mean_pred.cpu()
    mean_pred = mean_pred.detach().numpy()
    mean_pred = np.squeeze(mean_pred,0)
    #seq_pred = feats
    seq_pred = mean_pred
    ### write this sequence prediction to a csv file####
    total_seq_pred.append(seq_pred)

total_seq_pred = np.concatenate(total_seq_pred)

## For training and validation
"""
num_frames = transformed_dataset.n_frames    
csv_files = transformed_dataset.csv_file_name
"""

frames_count_file = os.path.join(base_dir,'Frames_Count.txt')    ### Frame count for Test dataset
## Load frame counts from txt file
file_writer = open(frames_count_file,'r')
csv_file=[]
frame_count =[]
for line in file_writer:
    row = str.split(line,' ')
    print(row)
    if (row[0]!= '\n'):
        csv_file.append(row[0])
        frame_count.append(row[2])

num_frames = frame_count
csv_files= csv_file

seq_frames = [num_frames[i]/25 for i in range(len(num_frames))]   ### number of 1 second frames
count = 0

for seq in range(len(seq_frames)):
    pred_frames = seq_frames[seq]
    seq_preds = total_seq_pred[count:pred_frames+count]

    assert (len(seq_preds) == pred_frames)
    count = pred_frames
    ## if some missing frames 
    final_seq_preds  = np.repeat(seq_preds,25,axis =0)

    csv_name = str.split(csv_files[seq], '/')
    csv_name = csv_name[len(csv_name)-1] 

    ## To save Visual Feature Embeddings
    """
    pred_embed_file = os.path.join(pred_feat_embed,(csv_name[0:len(csv_name)-4])+'.npy')
    np.save(pred_embed_file,seq_preds)
    """
    pred_csv_file = os.path.join(pred_output_file, (csv_name[0:len(csv_name)-4])+'.csv')

    with open(pred_csv_file, 'wb') as write:
        wr = csv.writer(write, lineterminator='\n')
        wr.writerow(['valence'])
        for row in final_seq_preds:
            wr.writerow([row])

### For Calculating Final CCC

"""
gt_target_dir = os.path.join(base_dir,'Annotations')
pred_output_dir = pred_output_file
meanccc = calculateCCC(gt_target_dir, pred_output_dir)

"""

### For calculating Avg CCC over Cross Validated Models ##


"""
results_path = os.path.join(base_dir,'Visual_Results', 'crossval.csv')
results_f = open(results_path, 'wb')
writer = csv.writer(results_f)
print("===")
print("Val. Story\tVal CCC")
writer.writerow(["Val. Story", "Val CCC"])
for story in range(len(val_stories)):
    print("{}\t\t{:0.3f}". \
          format(val_stories[story], val_ccc[story]))
    writer.writerow([val_stories[story],val_ccc[story]])
test_mean = sum(val_ccc) / len(val_ccc)
print("Average\t\t{:0.3f}".format(test_mean))
writer.writerow(["Average",test_mean])
results_f.close()
"""

