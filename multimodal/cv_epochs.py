"""Cross-validate models across epochs, assuming models are already saved."""

import argparse
import torch

import crossval

# Load parser from crossval.py and add options
parser = crossval.parser
parser.add_argument('--epoch_min', type=int, default=100,
                    help="Minimum epoch to evaluate.")
parser.add_argument('--epoch_max', type=int, default=2000,
                    help="Maximum epoch to evaluate.")
parser.add_argument('--epoch_step', type=int, default=100,
                    help="Step size when iterating over epochs.")

# Suppress help for overridden options
parser.add_argument('--test', help=argparse.SUPPRESS)
parser.add_argument('--features', help=argparse.SUPPRESS)
parser.add_argument('--test_epoch', help=argparse.SUPPRESS)

if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = args.mods.split(',')

    # Always test
    args.test = True
    args.features = False

    # Loop over epochs
    means, stds = dict(), dict()
    best_mean = 0
    for e in range(args.epoch_min, args.epoch_max+1, args.epoch_step):
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
