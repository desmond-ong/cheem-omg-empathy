"""Cross-validate by calling training script on different data."""

import os
import csv
import train
import torch

def main(args):
    # Extract flags that need to be overwritten for train.main
    test = args.test
    model_dir = args.model_dir
    split = args.split

    # Set constant flags to be passed on to train.main    
    args.load = None
    args.resume = False
    args.test_story = None

    # Load all data
    _, _, all_data = train.load_data(args.train_dir, args.test_dir)

    # Train and test with each story as the validation set
    train_ccc, test_ccc = dict(), dict()
    for story in sorted(list(set(all_data.stories))):
        print("---")
        print("Story {} as validation".format(story))
        print("---")
        
        # Split data
        test_data, train_data = all_data.extract_story([story])

        # Create new model directory
        args.model_dir = os.path.join(model_dir, "val_on_{}".format(story))

        # Load best model for story if test flag is set
        if test:
            args.load = os.path.join(args.model_dir, "best.save")
        
        # Compute best CCC on test set
        args.test = test
        args.split = split
        test_ccc[story] = train.main(train_data, test_data, args)

        # Test model on training set
        args.test = True
        args.split = 1
        train_ccc[story] = train.main(train_data, train_data, args)

    # Print and save results
    results_path = os.path.join(model_dir, "crossval.csv")
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
    parser.add_argument('--inputs', type=str, default="audio,text,v_sub,v_act",
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
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences (default: false)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--model_dir', type=str, default="./models",
                        help='path to save models')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()    
    main(args)

