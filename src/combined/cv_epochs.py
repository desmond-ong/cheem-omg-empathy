import argparse
import torch

import crossval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mods', type=str, default="audio,text,v_sub,v_act",
                        help='comma-separated input modalities (default: all)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: false)')
    parser.add_argument('--train_dir', type=str, default="./data/Training",
                        help='path to train data (default: ./data/Training)')
    parser.add_argument('--test_dir', type=str, default="./data/Validation",
                        help='path to test data (default: ./data/Validation)')
    parser.add_argument('--out_dir', type=str, default="./cv_models",
                        help='path to save models, predictions and results')
    parser.add_argument('--subjects', type=str, default=None, nargs='+',
                        help='subjects to train on (default: all)')
    parser.add_argument('--stories', type=str, default=None, nargs='+',
                        help='stories to train on (default: all)')

    # Default args to be passed on
    parser.add_argument('--diff', action='store_true', default=False,
                        help='whether to predict differences (default: false)')
    parser.add_argument('--recon', action='store_true', default=False,
                        help='whether to reconstruct inputs (default: false)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: false)')
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


    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = args.mods.split(',')

    # Always test
    args.test = True
    args.features = False

    # Loop over epochs
    means, stds = dict(), dict()
    best_mean = 0
    min_epochs, max_epochs, step = 100, 2000, 100
    for e in range(min_epochs, max_epochs+1, step):
        args.test_epoch = e
        means[e], stds[e] = crossval.main(args)
        if means[e] > best_mean:
            best_mean = means[e]
            best_e = e

    # Print mean CCC values
    print("===")
    print("Epoch CCC   STD  ")
    for e in sorted(means.keys()):
        print("{:5d} {:0.3f} {:0.3f}".format(e, means[e], stds[e]))
    print("---")
    print("Best epoch: {}".format(best_e))
    print("Best CCC: {:0.3f}".format(best_mean))
    print("===")

    # Save results and extract features for best epoch
    args.test_epoch = best_e
    args.features = True
    crossval.main(args)
