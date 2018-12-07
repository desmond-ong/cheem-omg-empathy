## Cross Validation Training script for OMG-Empathy Dataset ##

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
from data_loader import OMGEmpathyDataset
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from calculateCCC import ccc, mse, f1
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorboardX import SummaryWriter
from timer import Timer
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
#torch.cuda.set_device(1)

train_base_dir = './data/Training' 
val_base_dir = './data/Validation'   ##change the base directory for train, val and test
kwargs = {'num_workers': 4, 'pin_memory': True}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = os.path.join(train_base_dir, 'Avg_Visual_Features/Subject')
train_csv_dir = os.path.join(train_base_dir,'Annotations')
val_data_dir = os.path.join(val_base_dir, 'Avg_Visual_Features/Subject')
val_csv_dir = os.path.join(val_base_dir,'Annotations')
pattern = pattern = "Subject_(\d+)_Story_(\d+)(?:_\w*)?"

gd =20 ## gradient clipping parameter

### VGG_LSTM Model parameters
flag_biLSTM = False
lstm_input_size = 4096
h1 = 512 
output_dim=1
num_layers =1
loss_function = nn.MSELoss()


import re

class VGG_LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=1,use_cuda=True,feature=False):
        super(VGG_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.l1_dim = 128
        self.feature = feature
        #self.dropin = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)
        self.drop = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(self.hidden_dim, self.l1_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(self.l1_dim, self.output_dim)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device))

    def forward(self, input):
        hidden = self.init_hidden()
        input = input.view(self.batch_size,input.size(1),-1)
        lstm_out, hidden = self.lstm1(input)
        if flag_biLSTM:
            lstm_out = lstm_out.contiguous().view(self.batch_size, lstm_out.size(1), 2, -1).sum(2).view(self.batch_size,
                                                                                                            lstm_out.size(
                                                                                                             1), -1)
        drop_out = self.drop(lstm_out)
        ln1_out = self.linear1(drop_out)
        if self.feature:
            return ln1_out
        drop1_out = self.drop1(ln1_out)
        drop1_out= F.relu(drop1_out)
        y_pred = torch.tanh(self.linear2(drop1_out))
        return y_pred.view(self.batch_size, -1)

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r' + s)
    sys.stdout.flush()

def train(model,epoch,optimizer,train_dataloader,writer,log):

    timer = Timer()
    iter_timer = Timer()
    train_loss = 0
    batchid = 0
    model.train()
    n_iter = 0

    iter_timer.tic()
    for batch_idx, (data,target) in enumerate(train_dataloader):
        n_iter = n_iter+1
        timer.tic()
        data, target = Variable(data).cuda(), Variable(target).cuda()
        target = target.float()
        target = target.unsqueeze(1)

        seq_len = data.size(0)

        if (seq_len%batchsize == 0):
            final_seq = seq_len / batchsize
            target = target.view(batchsize,final_seq,-1)
            data = data.view(batchsize,final_seq,-1)
        else:
            rem = seq_len % batchsize
            data = data[0:seq_len-rem]
            target = target[0:seq_len-rem]
            final_seq = (data.size(0) / batchsize)
            target = target.view(batchsize,final_seq, -1)
            data = data.view(batchsize, final_seq, -1)
        target = target.squeeze(2)

        model.zero_grad()
        output = model(data)

      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

	## Test Gradient Clipping
        """
        if gd is not None:
            total_norm = clip_grad_norm(model.parameters(), gd)
            if total_norm > gd:
                print('clippling gradient: {} with coef {}'.format(total_norm, gd / total_norm))
        """
        target = target.cpu()
        target = torch.Tensor.numpy(target)
        target = np.squeeze(target)
        timer.toc()
        train_loss+=loss.data[0]
     
        if batch_idx % log== 0:
            printoneline('Epoch=%d Loss=%.4f\n'
                         % (epoch, train_loss / (batchid + 1)))

            writer.add_scalar('loss', train_loss, n_iter)
        batchid += 1
    iter_timer.toc()


def crossval_data(args,cv_stories,story):
    cv_data_dir = np.concatenate([[os.path.join(train_data_dir, fn) for fn in sorted(os.listdir(train_data_dir))],
                                  [os.path.join(val_data_dir, fn) for fn in sorted(os.listdir(val_data_dir))]], 0)
    cv_csv_dir = np.concatenate([[os.path.join(train_csv_dir, fn) for fn in sorted(os.listdir(train_csv_dir)) if
                                  re.match(pattern, fn) is not None],
                                 [os.path.join(val_csv_dir, fn) for fn in sorted(os.listdir(val_csv_dir)) if
                                  re.match(pattern, fn) is not None]], 0)


    train_cv_stories = np.delete(cv_stories,story)
    val_cv_stories = cv_stories[story]
    model_output = os.path.join('./models'+val_cv_stories)
    if not os.path.isdir(model_output):
        os.mkdirs(model_output)

    val_idx = [k for k, s in enumerate(cv_data_dir) if val_cv_stories in s]
    train_idx = []
    for j in range(len(train_cv_stories)):
        idx = [i for i, s in enumerate(cv_data_dir) if train_cv_stories[j] in s]
        train_idx.append(idx)
    train_idx=np.concatenate(train_idx)
    cv_train_data = np.take(cv_data_dir, train_idx)
    cv_train_csv = np.take(cv_csv_dir, train_idx)
    cv_val_data = np.take(cv_data_dir,val_idx)
    cv_val_csv = np.take(cv_csv_dir,val_idx)

    transformed_dataset = OMGEmpathyDataset(csv_dir=cv_train_csv, root_dir=cv_train_data, batch=batchsize)
    train_dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size * args.seq_len, shuffle=True, **kwargs)

    return model_output, train_dataloader


def main(args):
    cv_stories = ['Story_1', 'Story_2', 'Story_4', 'Story_5', 'Story_8'] 
    model = VGG_LSTM(lstm_input_size, h1, batch_size=args.batch_size, output_dim=output_dim, num_layers=num_layers,use_cuda=True)
    ## Cross Validation training ###
    for story in range(len(cv_stories)):
	    writer = SummaryWriter()
	    for epoch in range(1, args.epochs + 1):	       
	    	model_output, train_dataloader = crossval_data(args,cv_stories_story)
	    	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
	    	train(model,epoch,optimizer,train_dataloader,writer,args.log_interval)
		if epoch % 5 == 0:
		    output_file = os.path.join(model_output, "model_{}.pt".format(epoch))
		    print("Saving model to {}".format(output_file))
		    torch.save(model.state_dict(), output_file)
		"""
		if epoch % 1000 == 0:
		    args.lr = args.lr*0.1
		    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
		"""
		output_file = os.path.join(model_output, "model_{}.pt".format(epoch))
		torch.save(model.state_dict(), output_file)
   	    writer.close()


if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(description='OMG Empathy Training using visual features')
	parser.add_argument('--batch_size', type=int, default=16, metavar='N',
		            help='input batch size for training (default: 64)')
	parser.add_argument('--seq_len', type=int, default=150, metavar='N',
		            help='input sequence length for training (deafult=150 seconds)')
	parser.add_argument('--epochs', type=int, default=3000, metavar='N',
		            help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
		            help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
		            help='SGD momentum (default: 0.9)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
		            help='random seed (default: 1)')
	parser.add_argument('--log_interval', type=int, default=30, metavar='N',
		            help='how many batches to wait before logging training status')
	args = parser.parse_args()
	main(args)
