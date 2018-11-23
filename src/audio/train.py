"""Training code for audio modality"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil

import numpy as np
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
from model import AudioLSTM

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def train(loader, model, criterion, optimizer, epoch, args):
    data_num = 0
    loss= 0.0
    model.train()
    for batch_num, (data, target, lengths) in enumerate(loader):
        batch_size = data.size(0)
        if batch_size == 1:
            data, target = data.float(), target.float()
        # Compute differences for target
        if args.diff:
            target = target[:,1:] - target[:,:-1]
            target = torch.cat([torch.zeros(batch_size, 1, 1), target], dim=1)
        # Convert to CUDA
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Run forward pass.
        output = model(data, lengths)
        # Compute loss and gradients
        batch_loss = criterion(output, target)
        loss += batch_loss
        # Average over number of non-padding datapoints before stepping
        batch_loss /= sum(lengths)
        batch_loss.backward()
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(lengths)
        print('Batch: {:5d}\tLoss: {:2.5f}'.\
              format(batch_num, loss/data_num))
    # Average losses and print
    loss /= data_num
    print('---')
    print('Epoch: {}\tLoss: {:2.5f}'.format(epoch, loss))
    return loss

def evaluate(loader, model, criterion, args):
    seq_num, data_num = 0, 0
    loss, corr, ccc = 0.0, 0.0, 0.0
    model.eval()
    for batch_num, (data, target, lengths) in enumerate(loader):
        batch_size = data.size(0)
        if batch_size == 1:
            data, target = data.float(), target.float()
        # Compute differences for target
        if args.diff:
            diff = target[:,1:] - target[:,:-1]
            diff = torch.cat([torch.zeros(batch_size, 1, 1), diff], dim=1)
        # Convert to CUDA
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Run forward pass.
        output = model(data, lengths)
        # Compute loss and gradients
        if args.diff:
            batch_loss = criterion(output, diff)
        else:
            batch_loss = criterion(output, target)
        loss += batch_loss
        # Keep track of total number of time-points and sequences
        seq_num += batch_size
        data_num += sum(lengths)
        # Sum predicted differences
        if args.diff:
            output = torch.cumsum(output, dim=1)
        # Compute Pearson correlation and CCC for each sample
        for i in range(0, batch_size):
            y_true = target[i,:lengths[i]].view(-1)
            y_pred = output[i,:lengths[i]].view(-1)
            corr += pearsonr(y_true.cpu().numpy(), y_pred.cpu().numpy())[0]
            ccc += eval_ccc(y_true.cpu().numpy(), y_pred.cpu().numpy())
    # Average losses and print
    loss /= data_num
    corr /= seq_num
    ccc /= seq_num
    print('Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.3f}'.\
          format(loss, corr, ccc))
    return loss, corr, ccc

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda s, l: s)
    model.load_state_dict(checkpoint)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 250)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training')
    parser.add_argument('--model', type=str, default="./models/best.save",
                        help='path to trained model')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Load valence ratings and openSMILE features
    print("Loading data...")
    train_loader = DataLoader(datasets.OMGopenSMILE('./data/Training'),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.collate_fn)
    # Set batch_size=1 for evaluation
    test_loader = DataLoader(datasets.OMGopenSMILE('./data/Validation'),
                             batch_size=args.batch_size, shuffle=True,
                             collate_fn=datasets.collate_fn)
    print("Done.")
    
    # Create path to save models
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    # Construct audio-to-valence LSTM model
    input_size, hidden_size, num_layers = 990, 512, 1
    model = AudioLSTM(input_size, hidden_size, num_layers, args.cuda)

    # Setup loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        
    # Evaluate model if test flag is set
    if args.test:
        load_checkpoint(model, args.model, args.cuda)
        with torch.no_grad():
            evaluate(test_loader, model, criterion, args)
        sys.exit(0)
    
    # Otherwise train and save best model
    best_ccc = -2
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, criterion, optimizer, epoch, args)
        with torch.no_grad():
            loss, corr, ccc = evaluate(test_loader, model, criterion, args)
        if ccc > best_ccc:
            best_ccc = ccc
            save_checkpoint(model, "./models/best.save")
        if epoch % args.save_freq == 0:
            path = os.path.join("./models",
                                "epoch_{}.save".format(epoch)) 
            save_checkpoint(model, path)
