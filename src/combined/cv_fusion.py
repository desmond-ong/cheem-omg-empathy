""""Cross-validation for decision-level fusion."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, shutil, re
import joblib
import pandas as pd
import numpy as np

import fusion
from datasets import OMGFusion

def copy_all(src, dst, pattern=".*"):
    """Copies all contents of src that match pattern to dst."""
    if not os.path.exists(dst):
        os.makedirs(dst)
    for fn in os.listdir(src):
        if not re.match(pattern, fn):
            continue
        shutil.copy(os.path.join(src, fn), dst)

def main(args):
    # Extract flags that need to be overwritten for fusion.main
    test = args.test
    
    # Set constant flags to be passed on to fusion.main
    args.test_story = None

    # Create folder to save predictions on validation sets
    cv_pred_dir = os.path.join(args.out_dir, "cv_pred")

    # Load all valence ratings so we know which stories are in the dataset
    val_path = os.path.join(args.in_dir,"Annotations")
    all_data = OMGFusion([], [], val_path)
    
    # Train and test with each story as the validation set
    train_ccc, test_ccc = dict(), dict()
    for story in sorted(list(set(all_data.stories))):
        print("---")
        print("Story {} as validation".format(story))

        # Copy features from crossval.py output directories
        for cv_dir, in_name in zip(args.cv_dirs, args.in_names):
            story_dir = os.path.join(cv_dir, "val_on_{}".format(story))
            copy_all(os.path.join(story_dir, "feat_train"),
                     os.path.join(args.in_dir, in_name), "^.*\.npy")
            copy_all(os.path.join(story_dir, "feat_test"),
                     os.path.join(args.in_dir, in_name), "^.*\.npy")

        # Load all features
        in_dirs = [os.path.join(args.in_dir, n) for n in args.in_names]
        all_data = OMGFusion(args.in_names, in_dirs, val_path,
                             normalize=args.normalize)

        # Split data
        test_data, train_data = all_data.extract_story([story])        
            
        # Create new subdirectory to store models and predictions for story
        story_dir = os.path.join(args.out_dir, "val_on_{}".format(story))
        args.model_dir = story_dir
        
        # Fit model if test flag is not set
        if not test:
            args.test = None
            args.pred_dir = story_dir
            print("---")
            fusion.main(train_data, test_data, args)

        # Load best model for story if test flag is set
        args.test = os.path.join(args.model_dir, "best.save")
            
        # Evaluate model on test and training set
        print("---")
        print("Evaluating best model...")
        print("---")
        args.pred_dir = story_dir
        ccc1, ccc2 = fusion.main(train_data, test_data, args)
        train_ccc[story] = ccc1
        test_ccc[story] = ccc2
        
        # Collate predictions on test set
        pred_test_dir = os.path.join(args.pred_dir, "pred_test")
        copy_all(pred_test_dir, cv_pred_dir, "^.*\.csv")

        # Delete features that were copied
        for in_name in args.in_names:
            shutil.rmtree(os.path.join(args.in_dir, in_name))
            
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
    train_mean = sum(train_ccc.values()) / len(train_ccc)
    test_mean = sum(test_ccc.values()) / len(test_ccc)
    print("Average\t\t{:0.3f}\t\t{:0.3f}".format(train_mean, test_mean))
    writer.writerow(["Average", train_mean, test_mean])
    results_f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cv_dirs', type=str, nargs='+', metavar='DIR',
                        help='paths to base folders generated by crossval.py')
    parser.add_argument('--in_names', type=str, nargs='+', metavar='NAME',
                        help='names for input features')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: True)')
    parser.add_argument('--test', type=str, default=None,
                        help='path to model to test (default: None)')
    parser.add_argument('--in_dir', type=str, default="./data/All/",
                        help='base folder for all data')
    parser.add_argument('--out_dir', type=str, default="./cv_fusion_out",
                        help='path to save models, predictions and results')
    args = parser.parse_args()
    if args.in_names is None:
        args.in_names = [os.path.basename(d).lower() for d in args.cv_dirs]
    main(args)
