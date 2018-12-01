"""Cross-validate by calling training script on different data."""

import os
import csv
import train

def main(args):
    # Set constant flags to be passed on to train.main
    args.resume = False
    args.load = None
    model_dir = args.model_dir

    # Load all data
    _, _, all_data = train.load_data(args.train_dir, args.test_dir)

    # Train and test with each story as the validation set
    train_ccc, test_ccc = dict(), dict()
    for story in set(all_data.stories):
        test_data, train_data = all_data.extract_story([story])

        # Creat new model directory
        args.model_dir = os.path.join(model_dir, "val_on_{}".format(story))
        
        # Train and compute best CCC on test set
        args.test = False
        test_ccc[story] = main(train_data, test_data, args)

        # Test model on training set
        args.test = True
        train_ccc[story] = main(train_data, train_data, args)

    # Print and save results
    results_path = os.join(model_dir, "crossval.csv")
    results_f = open(results_path, 'wb')
    writer = csv.writer(results_f)
    print("===")
    print("Val. Story\tTrain CCC\tVal. CCC")
    writer.writerow(["Val. Story", "Train CCC", "Val. CCC"])
    for story in train_ccc.keys():
        print("{:i}\t{:0.3f}\t{0.3f}".\
              format(story, train_ccc[story], test_ccc[story]))
        writer.writerow([story, train_ccc[story], test_ccc[story])
    reults_f.close()
        
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
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences (default: false)')
    parser.add_argument('--test_story', type=int, default=None,
                        help='story to use as test set (optional)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--model_dir', type=str, default="./models",
                        help='path to save models')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()    
    main(args)

