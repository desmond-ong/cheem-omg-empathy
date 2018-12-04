"""Cross-validate by calling training script on different data."""

import os, re, shutil
import csv
import train
import torch

def main(args):
    # Extract flags that need to be overwritten for train.main
    test = args.test
    features = args.features
    split = args.split

    # Set constant flags to be passed on to train.main    
    args.load = None
    args.resume = False
    args.test_story = None

    # Create folder to save predictions on validation sets
    cv_pred_dir = os.path.join(args.out_dir, "cv_pred")
    if not os.path.exists(cv_pred_dir):
        os.makedirs(cv_pred_dir)
    
    # Load all data
    _, _, all_data = train.load_data(args.mods, args.train_dir, args.test_dir)

    # Train and test with each story as the validation set
    train_ccc, test_ccc = dict(), dict()
    for story in sorted(list(set(all_data.stories))):
        print("---")
        print("Story {} as validation".format(story))
        
        # Split data
        test_data, train_data = all_data.extract_story([story])

        # Create new subdirectory to store models and predictions
        story_dir = os.path.join(args.out_dir, "val_on_{}".format(story))
        args.model_dir = story_dir
        args.pred_dir = story_dir
        args.feat_dir = story_dir
        
        # Train model
        if not (test or features):
            args.split = split
            args.test, args.features = False, False
            print("---")
            train.main(train_data, test_data, args)

        # Load best model for story
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
    train_mean = sum(train_ccc.values()) / len(train_ccc)
    test_mean = sum(test_ccc.values()) / len(test_ccc)
    print("Average\t\t{:0.3f}\t\t{:0.3f}".format(train_mean, test_mean))
    writer.writerow(["Average", train_mean, test_mean])
    results_f.close()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
                        help='evaluate after this many epochs (default: 1)')
    parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                        help='save model after this many epochs (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: false)')
    parser.add_argument('--recon', action='store_true', default=False,
                        help='whether to reconstruct inputs (default: false)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: false)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--features', action='store_true', default=False,
                        help='extract features from model (default: false)')
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences (default: false)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--out_dir', type=str, default="./cv_models",
                        help='path to save models, predictions and results')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = args.mods.split(',')
    main(args)

