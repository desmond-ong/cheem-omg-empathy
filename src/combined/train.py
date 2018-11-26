"""Training code for audio modality"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
from model import CombinedLSTM

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
    for batch_num, batch in enumerate(loader):
        batch = list(batch)
        batch_size = len(batch[-1])
        # Convert to float
        if batch_size == 1:
            for i in range(len(batch)-1):
                batch[i] = batch[i].float()
        # Convert to CUDA
        if args.cuda:
            for i in range(len(batch)-1):
                batch[i] = batch[i].cuda()
        # Unpack batch
        audio, text, visual, target, lengths = batch
        # Compute differences for target
        if args.diff:
            target = target[:,1:] - target[:,:-1]
            target = torch.cat([torch.zeros(batch_size, 1, 1), target], dim=1)
        # Run forward pass.
        output = model(audio, text, visual, lengths)
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
    pred = []
    seq_num, data_num = 0, 0
    loss, corr, ccc = 0.0, 0.0, 0.0
    model.eval()
    for batch_num, batch in enumerate(loader):
        batch = list(batch)
        batch_size = len(batch[-1])
        # Convert to float
        if batch_size == 1:
            for i in range(len(batch)-1):
                batch[i] = batch[i].float()
        # Convert to CUDA
        if args.cuda:
            for i in range(len(batch)-1):
                batch[i] = batch[i].cuda()
        # Unpack batch
        audio, text, visual, target, lengths = batch
        # Run forward pass.
        output = model(audio, text, visual, lengths)
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
        # Store predictions
        for i in range(0, batch_size):
            pred_i = output[i,:lengths[i]].view(-1).cpu().numpy()
            pred.append(pred_i)
    # Compute CCC of predictions with original unsmoothed data
    time_ratio = loader.dataset.time_ratio
    true = loader.dataset.valence_orig
    for i in range(len(pred)):
        # Repeat and pad predictions to match original data length
        pred[i] = np.repeat(pred[i], time_ratio)[:len(true[i])]
        l_diff = len(true[i]) - len(pred[i])
        if l_diff > 0:
            pred[i] = np.concatenate([pred[i], pred[i][-l_diff:]])
        corr += pearsonr(true[i], pred[i])[0]
        ccc += eval_ccc(true[i], pred[i])
    # Average losses and print
    loss /= data_num
    corr /= seq_num
    ccc /= seq_num
    print('Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.3f}'.\
          format(loss, corr, ccc))
    return pred, loss, corr, ccc

def save_predictions(pred, dataset):
    for p, subj, story in zip(pred, dataset.subjects, dataset.stories):
        df = pd.DataFrame(p, columns=['valence'])
        fname = "Subject_{}_Story_{}.csv".format(subj, story)
        df.to_csv(os.path.join("./predictions", fname), index=False)

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
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 5)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training')
    parser.add_argument('--test_path', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--model', type=str, default="./models/best.save",
                        help='path to trained model')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Load data
    print("Loading data...")
    train_folder = "./data/Training"
    train_data = datasets.OMGcombined(
        os.path.join(train_folder,"CombinedAudio"),
        os.path.join(train_folder,"CombinedText"),
        os.path.join(train_folder,"CombinedVisual"),
        os.path.join(train_folder,"Annotations"),
        split_ratio=args.split
    )
    test_folder = args.test_path
    test_data = datasets.OMGcombined(
        os.path.join(test_folder,"CombinedAudio"),
        os.path.join(test_folder,"CombinedText"),
        os.path.join(test_folder,"CombinedVisual"),
        os.path.join(test_folder,"Annotations")
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=datasets.collate_fn)
    test_loader = DataLoader(test_data, batch_size=1,
                             shuffle=False, collate_fn=datasets.collate_fn)
    print("Done.")
    
    # Create path to save models and predictions
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')
    
    # Construct audio-text-visual LSTM model
    model = CombinedLSTM(use_cuda=args.cuda)

    # Setup loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        
    # Evaluate model if test flag is set
    if args.test:
        load_checkpoint(model, args.model, args.cuda)
        with torch.no_grad():
            pred, _, _, _ = evaluate(test_loader, model, criterion, args)
        save_predictions(pred, test_data)
        sys.exit(0)
    
    # Otherwise train and save best model
    best_ccc = -2
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, criterion, optimizer, epoch, args)
        with torch.no_grad():
            pred, loss, corr, ccc =\
                evaluate(test_loader, model, criterion, args)
        if ccc > best_ccc:
            best_ccc = ccc
            save_checkpoint(model, "./models/best.save")
        if epoch % args.save_freq == 0:
            path = os.path.join("./models",
                                "epoch_{}.save".format(epoch)) 
            save_checkpoint(model, path)
