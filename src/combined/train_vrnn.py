"""Training code for VRNN model."""

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
from vrnn import VRNN

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def train(loader, model, optimizer, epoch, args):
    data_num = 0
    loss= 0.0
    model.train()
    # Iterate over batches
    for batch_num, batch in enumerate(loader):
        batch = list(batch)
        batch_size = len(batch[-1])
        # Transform inputs
        for i in range(len(batch)-1):
            # Convert to float
            if batch_size == 1:
                batch[i] = batch[i].float()
            # Convert to CUDA
            if args.cuda:
                batch[i] = batch[i].cuda()
            # Permute so time dimension is first
            batch[i] = batch[i].permute(1, 0, 2)
        # Unpack batch
        audio, text, visual, val_obs, lengths = batch
        # Run forward pass.
        infer, prior, recon, val = model(audio, text, visual)
        # Create mask from sequence lengths
        mask = datasets.len_to_mask(lengths).permute(1, 0).unsqueeze(-1)
        if args.cuda:
            mask = mask.cuda()
        # Compute loss and gradients
        inputs = (audio, text, visual)
        b_loss = model.loss(inputs, val_obs, infer, prior, recon, val, mask,
                            args.kld_mult, args.rec_mults, args.sup_mult)
        loss += b_loss
        # Average over number of datapoints before stepping
        b_loss /= sum(lengths)
        b_loss.backward()
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(lengths)
        print('Batch: {:5d}\tLoss: {:10.1f}'.\
              format(batch_num, loss/data_num))
    # Average losses and print
    loss /= data_num
    print('---')
    print('Epoch: {}\tLoss: {:10.1f}'.format(epoch, loss))
    return loss

def evaluate(loader, model, args):
    pred = []
    seq_num, data_num = 0, 0
    kld_loss, rec_loss, sup_loss = 0.0, 0.0, 0.0
    mse, corr, ccc = 0.0, 0.0, 0.0
    model.eval()
    for batch_num, batch in enumerate(loader):
        batch = list(batch)
        batch_size = len(batch[-1])
        # Transform inputs
        for i in range(len(batch)-1):
            # Convert to float
            if batch_size == 1:
                batch[i] = batch[i].float()
            # Convert to CUDA
            if args.cuda:
                batch[i] = batch[i].cuda()
            # Permute so time dimension is first
            batch[i] = batch[i].permute(1, 0, 2)
        # Unpack batch
        audio, text, visual, val_obs, lengths = batch
        # Run forward pass
        infer, prior, recon, val = model(audio, text, visual)
        val_mean, val_std = val
        # Create mask from sequence lengths
        mask = datasets.len_to_mask(lengths).permute(1, 0).unsqueeze(-1)
        if args.cuda:
            mask = mask.cuda()
        # Compute and store losses
        inputs = (audio, text, visual)
        kld_loss += model.kld_loss(infer, prior, mask)
        rec_loss += model.rec_loss(inputs, recon, mask, args.rec_mults)
        sup_loss += model.sup_loss(val, val_obs, mask)
        # Keep track of total number of time-points and sequences
        seq_num += batch_size
        data_num += sum(lengths)
        # Store predictions
        for i in range(0, batch_size):
            pred_i = val_mean[:lengths[i],i].view(-1).cpu().numpy()
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
        mse += ((true[i]-pred[i])**2).sum()
        corr += pearsonr(true[i], pred[i])[0]
        ccc += eval_ccc(true[i], pred[i])
    # Average losses and print
    kld_loss /= data_num
    rec_loss /= data_num
    sup_loss /= data_num
    losses = kld_loss, rec_loss, sup_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\tSup: {:7.1f}'.\
          format(*losses))
    # Average statistics and print
    mse /= sum([len(t) for t in true])
    corr /= seq_num
    ccc /= seq_num
    stats = mse, corr, ccc
    print('\t\tMSE: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.3f}'.format(*stats))
    return pred, losses, stats

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
    parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 25)')
    parser.add_argument('--split', type=int, default=5, metavar='N',
                        help='sections to split each video into (default: 5)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='how many epochs to wait before saving')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: false)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training loaded model (default: false)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
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
    
    # Construct multi-modal VRNN model
    model = VRNN(use_cuda=args.cuda)

    # Setup optimizer and loss multipliers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.kld_mult = 1.0
    args.sup_mult = 1e5
    args.rec_mults = (1/model.audio_dim, 1/model.text_dim, 1/model.visual_dim)
        
    # Evaluate model if test flag is set
    if args.test:
        load_checkpoint(model, args.model, args.cuda)
        with torch.no_grad():
            pred, _, _ = evaluate(test_loader, model, args)
        save_predictions(pred, test_data)
        sys.exit(0)

    # Load model if continue flag is set
    if args.resume:
        load_checkpoint(model, args.model, args.cuda)
        
    # Train and save best model
    best_ccc = -2
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, optimizer, epoch, args)
        with torch.no_grad():
            pred, losses, stats = evaluate(test_loader, model, args)
            mse, corr, ccc = stats
        if ccc > best_ccc:
            best_ccc = ccc
            save_checkpoint(model, "./models/best.save")
        if epoch % args.save_freq == 0:
            path = os.path.join("./models",
                                "epoch_{}.save".format(epoch)) 
            save_checkpoint(model, path)
