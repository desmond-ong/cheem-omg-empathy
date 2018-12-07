"""Cross-validate by calling training script on different data."""

import os, re, shutil, csv
import argparse
import numpy as np
import torch

import train

def main(args):
    # Extract flags that need to be overwritten for train.main
    test = args.test
    features = args.features
    split = args.split

    # Create folder to save predictions on validation sets
    cv_pred_dir = os.path.join(args.out_dir, "cv_pred")
    if not os.path.exists(cv_pred_dir):
        os.makedirs(cv_pred_dir)
    
    # Load all data
    _, _, all_data = train.load_data(args.mods, args.train_dir, args.test_dir)
    
    # Keep only specified stories and subjects
    all_data, _ = all_data.extract(args.stories, args.subjects)

    # Normalize inputs
    if args.normalize:
        all_data.normalize()
    
    # Train and test with each story as the validation set
    train_ccc, test_ccc = dict(), dict()
    for story in sorted(list(set(all_data.stories))):
        print("---")
        print("Story {} as validation".format(story))
        
        # Split data
        test_data, train_data = all_data.extract(stories=[story])

        # Create new subdirectory to store models and predictions
        story_dir = os.path.join(args.out_dir, "val_on_{}".format(story))
        args.model_dir = story_dir
        args.pred_dir = story_dir
        args.feat_dir = story_dir
        
        # Train model
        if not (test or features):
            args.split = split
            args.load = None
            args.test, args.features = False, False
            print("---")
            train.main(train_data, test_data, args)

        # Load model to test and extract features (default to best model)
        if args.test_epoch is not None:
            model_fname = "epoch_{}.save".format(args.test_epoch)
            args.load = os.path.join(args.model_dir, model_fname)
        else:
            args.load = os.path.join(args.model_dir, "best.save")
            
        # Test model on test and training set
        print("---")
        print("Evaluating best model...")
        print("---")
        args.split = 1
        args.test, args.features = True, False
        ccc1, ccc2 = train.main(train_data, test_data, args)
        train_ccc[story] = ccc1
        test_ccc[story] = ccc2
        
        # Collate predictions on test set
        pred_test_dir = os.path.join(args.pred_dir, "pred_test")
        for fn in os.listdir(pred_test_dir):
            if not re.match("^.*\.csv", fn):
                continue
            shutil.copy(os.path.join(pred_test_dir, fn), cv_pred_dir)

        # Extract features on test and training set
        if features:
            print("---")
            print("Extracting features...")
            args.test, args.features = False, True
            train.main(train_data, test_data, args)
            
    # Print and save results
    results_path = os.path.join(args.out_dir, "crossval.csv")
    results_f = open(results_path, 'wb')
    writer = csv.writer(results_f)

    print("===")
    print("Val. Story\tTrain CCC\tVal. CCC")    
    writer.writerow(["Val. Story", "Train CCC", "Val. CCC"])
    
    for story in sorted(train_ccc.keys()):
        print("{}\t\t{:0.3f}\t\t{:0.3f}".\
              format(story, train_ccc[story], test_ccc[story]))
        writer.writerow([story, train_ccc[story], test_ccc[story]])

    # Compute mean, std, and mean minus std
    train_mean = np.mean(train_ccc.values())
    test_mean = np.mean(test_ccc.values())
    train_std = np.std(train_ccc.values())
    test_std = np.std(test_ccc.values())

    print("Mean\t\t{:0.3f}\t\t{:0.3f}".format(train_mean, test_mean))
    print("Std\t\t{:0.3f}\t\t{:0.3f}".format(train_std, test_std))
    print("M-S\t\t{:0.3f}\t\t{:0.3f}".format(train_mean-train_std,
                                             test_mean-test_std))
    writer.writerow(["Mean", train_mean, test_mean])
    writer.writerow(["Std", train_std, test_std])
    writer.writerow(["M-S", train_mean-train_std, test_mean-test_std])
    results_f.close()

    return test_mean, test_std

# Load parser from train.py and add options
parser = train.parser
parser.add_argument('--subjects', type=int, default=None, nargs='+',
                    help='subjects to train on (default: all)')
parser.add_argument('--stories', type=int, default=None, nargs='+',
                    help='stories to train on (default: all)')
parser.add_argument('--out_dir', type=str, default="./cv_models",
                    help='path to save models, predictions and results')
parser.add_argument('--test_epoch', type=int, default=None, metavar='N',
                    help='epoch to cross-validate (default: best)')

# Suppress help for overridden options
parser.add_argument('--model_dir', help=argparse.SUPPRESS)
parser.add_argument('--pred_dir', help=argparse.SUPPRESS)
parser.add_argument('--feat_dir', help=argparse.SUPPRESS)
parser.add_argument('--resume', help=argparse.SUPPRESS)
parser.add_argument('--load', help=argparse.SUPPRESS)

if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = args.mods.split(',')
    main(args)

