"""Training code for combined LSTM model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse

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
        # Convert tensors to float
        if batch_size == 1:
            for i in range(len(batch)-1):
                batch[i] = batch[i].float()
        # Convert tensors to CUDA
        if args.cuda:
            for i in range(len(batch)-1):
                batch[i] = batch[i].cuda()
        # Unpack batch
        inputs = dict(zip(args.mods, batch[:-3]))
        target = batch[-3]
        mask, lengths = batch[-2:]
        # Compute differences for target
        if args.diff:
            target = target[:,1:] - target[:,:-1]
            target = torch.cat([torch.zeros(batch_size, 1, 1), target], dim=1)
        # Run forward pass.
        output, recon = model(inputs, mask, lengths)
        # Compute loss and gradients
        batch_loss = criterion(output, target)
        # Compute reconstruction loss if recon flag is set
        if args.recon:
            for m in model.mods:
                # MSE loss between reconstruction at t and input at t+1
                m_loss = criterion(recon[m][:,:-1], inputs[m][:,1:])
                # Divide loss by modality dims and number to keep balance
                batch_loss += 0.05*m_loss / (model.dims[m] * len(model.mods))
        # Accumulate total loss for epoch
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

def evaluate(dataset, model, criterion, args):
    pred = []
    data_num = 0
    loss, corr, ccc = 0.0, 0.0, 0.0
    model.eval()
    for data in dataset:
        # Collate data into batch of size 1
        batch = list(datasets.collate_fn([data]))
        # Convert to float
        for i in range(len(batch)-1):
            batch[i] = batch[i].float()
        # Convert to CUDA
        if args.cuda:
            for i in range(len(batch)-1):
                batch[i] = batch[i].cuda()
        # Unpack batch
        inputs = dict(zip(args.mods, batch[:-3]))
        target = batch[-3]
        mask, lengths = batch[-2:]
        # Run forward pass.
        output, recon = model(inputs, mask, lengths)
        # Compute loss (only for target)
        if args.diff:
            loss += criterion(output, diff)
        else:
            loss += criterion(output, target)
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Sum predicted differences
        if args.diff:
            output = torch.cumsum(output, dim=1)
        # Store predictions
        pred_i = output[0,:lengths[0]].view(-1).cpu().numpy()
        pred.append(pred_i)
    # Compute CCC of predictions with original unsmoothed data
    time_ratio = dataset.time_ratio
    true = dataset.val_orig
    for i in range(len(pred)):
        # Repeat and pad predictions to match original data length
        pred[i] = np.repeat(pred[i], dataset.time_ratio)[:len(true[i])]
        l_diff = len(true[i]) - len(pred[i])
        if l_diff > 0:
            pred[i] = np.concatenate([pred[i], pred[i][-l_diff:]])
        corr += pearsonr(true[i], pred[i])[0]
        ccc += eval_ccc(true[i], pred[i])
    # Average losses and print
    loss /= data_num
    corr /= len(dataset)
    ccc /= len(dataset)
    print('Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.3f}'.\
          format(loss, corr, ccc))
    return pred, loss, corr, ccc

def save_features(dataset, model, path, args):
    model.eval()
    for data, subj, story in zip(dataset, dataset.subjects, dataset.stories):
        # Collate data into batch of size 1
        batch = list(datasets.collate_fn([data]))
        # Convert to float
        for i in range(len(data)-1):
            batch[i] = batch[i].float()
        # Convert to CUDA
        if args.cuda:
            for i in range(len(batch)-1):
                batch[i] = batch[i].cuda()
        # Unpack batch
        inputs = dict(zip(args.mods, batch[:-3]))
        target = batch[-3]
        mask, lengths = batch[-2:]
        # Run forward pass.
        features = model(inputs, mask, lengths, output_features=True)
        features = features.squeeze(0).cpu().numpy()
        # Save features to NPY files
        fname = "Subject_{}_Story_{}.npy".format(subj, story)
        np.save(os.path.join(path, fname), features)

def save_predictions(dataset, pred, path):
    for p, subj, story in zip(pred, dataset.subjects, dataset.stories):
        df = pd.DataFrame(p, columns=['valence'])
        fname = "Subject_{}_Story_{}.csv".format(subj, story)
        df.to_csv(os.path.join(path, fname), index=False)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda s, l: s)
    model.load_state_dict(checkpoint)

def load_data(modalities, train_dir, test_dir):
    print("Loading data...")
    dirs = {'audio': "CombinedAudio", 'text': "CombinedText",
            'v_sub': "CombinedVSub", 'v_act': "CombinedVAct"}
    train_data = datasets.OMGMulti(
        modalities, [os.path.join(train_dir, dirs[m]) for m in modalities],
        os.path.join(train_dir,"Annotations"), truncate=True)
    test_data = datasets.OMGMulti(
        modalities, [os.path.join(test_dir, dirs[m]) for m in modalities],
        os.path.join(test_dir,"Annotations"))
    all_data = train_data.join(test_data)
    print("Done.")
    return train_data, test_data, all_data

def preprocess_data(args):
    # Load data
    train_data, test_data, all_data =\
        load_data(args.mods, args.train_dir, args.test_dir)
    
    # Normalize inputs
    if args.normalize:
        all_data.normalize()
        
    # Make new train/test split
    if args.test_set is not None:
        test_data, train_data = all_data.extract(stories=args.test_set)

    # Extract/augment subjects
    if args.subjects is not None:
        # Test set should only contain specified subjects
        test_data, _ = test_data.extract(subjects=args.subjects)
        if args.augment is not None:
            # Augment training set if augment flag is set
            train_data = train_data.augment(subjects=args.subjects,
                                            mult=args.augment)
        else:
            # Otherwise training set should contain only specified subjects
            train_data, _ = train_data.extract(subjects=args.subjects)

    return train_data, test_data

def main(train_data, test_data, args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    
    # Construct audio-text-visual LSTM model
    dims = {'audio': 990, 'text': 300, 'v_sub': 4096, 'v_act': 4096}
    model = CombinedLSTM(mods=args.mods, dims=(dims[m] for m in args.mods),
                         reconstruct=args.recon, use_cuda=args.cuda)

    # Setup loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Load model if resume, test, or feature flags are set
    if args.test or args.features or args.resume:
        if args.load is not None:
            load_checkpoint(model, args.load, args.cuda)
        else:
            # Load best model in model dir if unspecified
            model_path = os.path.join(args.model_dir, "best.save")
            load_checkpoint(model, model_path, args.cuda)
    
    # Evaluate model if test flag is set
    if args.test:
        # Create paths to save features
        pred_train_dir = os.path.join(args.pred_dir, "pred_train")
        pred_test_dir = os.path.join(args.pred_dir, "pred_test")
        if not os.path.exists(pred_train_dir):
            os.makedirs(pred_train_dir)
        if not os.path.exists(pred_test_dir):
            os.makedirs(pred_test_dir)
        # Evaluate on both training and test set
        with torch.no_grad():
            pred, _, _, ccc1 = evaluate(train_data, model, criterion, args)
            save_predictions(train_data, pred, pred_train_dir)
            pred, _, _, ccc2 = evaluate(test_data, model, criterion, args)
            save_predictions(test_data, pred, pred_test_dir)
        return ccc1, ccc2

    # Save features if flag is set
    if args.features:
        # Create paths to save features
        feat_train_dir = os.path.join(args.feat_dir, "feat_train")
        feat_test_dir = os.path.join(args.feat_dir, "feat_test")
        if not os.path.exists(feat_train_dir):
            os.makedirs(feat_train_dir)
        if not os.path.exists(feat_test_dir):
            os.makedirs(feat_test_dir)
        # Save features for both training and test set
        with torch.no_grad():
            save_features(train_data, model, feat_train_dir, args)
            save_features(test_data, model, feat_test_dir, args)
        return

    # Split training data into chunks
    train_data.split(args.split)
    # Batch data using data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=datasets.collate_fn)

    # Create path to save models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Train and save best model
    best_ccc = -2
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                pred, loss, corr, ccc =\
                    evaluate(test_data, model, criterion, args)
            if ccc > best_ccc:
                best_ccc = ccc
                path = os.path.join(args.model_dir, "best.save") 
                save_checkpoint(model, path)
        # Save checkpoints
        if epoch % args.save_freq == 0:
            path = os.path.join(args.model_dir,
                                "epoch_{}.save".format(epoch)) 
            save_checkpoint(model, path)

    # Save final model
    path = os.path.join(args.model_dir, "last.save") 
    save_checkpoint(model, path)
    
    # Unsplit training data before returning
    train_data.split(1)
    
    return best_ccc

# Define parser as global variable so it can be re-used in imports
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--mods', type=str, default="audio,text,v_sub,v_act",
                    help='comma-separated input modalities (default: all)')
parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                    help='input batch size for training (default: 25)')
parser.add_argument('--split', type=int, default=5, metavar='N',
                    help='sections to split each video into (default: 5)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1e-5)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                    help='evaluate every N epochs (default: 1)')
parser.add_argument('--save_freq', type=int, default=100, metavar='N',
                    help='save every N epochs (default: 100)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: false)')
parser.add_argument('--diff', action='store_true', default=False,
                    help='whether to predict differences (default: false)')
parser.add_argument('--recon', action='store_true', default=False,
                    help='whether to reconstruct inputs (default: false)')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='whether to normalize inputs (default: false)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training loaded model (default: false)')
parser.add_argument('--features', action='store_true', default=False,
                    help='extract features from model (default: false)')
parser.add_argument('--test', action='store_true', default=False,
                    help='evaluate without training (default: false)')
parser.add_argument('--test_set', type=int, default=None, nargs='+',
                    help='stories to use as test set (optional)')
parser.add_argument('--subjects', type=int, default=None, nargs='+',
                    help='subjects to train on or emphasize (optional)')
parser.add_argument('--augment', type=int, default=None, metavar='N',
                    help='augmentation ratio for subjects (optional)')
parser.add_argument('--load', type=str, default=None,
                    help='path to load trained model')
parser.add_argument('--train_dir', type=str, default="./data/Training",
                    help='path to train data (default: ./data/Training)')
parser.add_argument('--test_dir', type=str, default="./data/Validation",
                    help='path to test data (default: ./data/Validation)')
parser.add_argument('--model_dir', type=str, default="./models",
                    help='path to save models')
parser.add_argument('--pred_dir', type=str, default="./",
                    help='path to save predictions')
parser.add_argument('--feat_dir', type=str, default="./",
                    help='path to save extracted features')

if __name__ == "__main__":
    # Parse command line args
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = tuple(args.mods.split(','))

    # Load and preprocess data
    train_data, test_data = preprocess_data(args)
    
    # Continue to rest of script
    main(train_data, test_data, args)
