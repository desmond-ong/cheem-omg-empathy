"""Fine-tune a general model for each subject."""

import os
import argparse
import torch

import train

# Load parser from train.py and add options
parser = train.parser
parser.add_argument('--out_dir', type=str, default="./personal",
                    help='base directory to save personalized models')

# Suppress help for overridden options
parser.add_argument('--model_dir', help=argparse.SUPPRESS)
parser.add_argument('--pred_dir', help=argparse.SUPPRESS)
parser.add_argument('--feat_dir', help=argparse.SUPPRESS)
parser.add_argument('--resume', help=argparse.SUPPRESS)
parser.add_argument('--subjects', help=argparse.SUPPRESS)

if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.mods = tuple(args.mods.split(','))
    args.pred_dir = args.out_dir

    # Extract args that will be overridden
    test = args.test
    
    # Load and preprocess data
    train_data, test_data, all_data =\
        train.load_data(args.mods, args.train_dir, args.test_dir)
    if args.normalize:
        all_data.normalize()
    test_data, train_data = all_data.extract(stories=args.test_set)
    
    # Fine-tune model for each subject
    subjects = list(sorted(list(set(all_data.subjects))))
    subj_ccc = []
    for s in subjects:    
        # Test only on current subject
        test_data_s, _ = test_data.extract(subjects=[s])
        if args.augment is not None:
            # Augment training set if augment flag is set
            train_data_s = train_data.augment(subjects=[s], mult=args.augment)
        else:
            # Otherwise training set should contain only current subject
            train_data_s, _ = train_data.extract(subjects=[s])

        # Create subdirectory for subject
        subj_dir = os.path.join(args.out_dir, "subject_{}".format(s))
        args.model_dir = subj_dir
        args.feat_dir = subj_dir

        # Load pretrained model to fine-tune
        args.load = os.path.join(args.out_dir, "init.save")
        
        # Train model
        if not test:
            args.test, args.resume = False, True
            train.main(train_data_s, test_data_s, args)

        # Load and evaluate model from last epoch
        args.load = os.path.join(subj_dir, "last.save")
        args.test, args.resume = True, False
        train_ccc, test_ccc = train.main(train_data_s, test_data_s, args)

        # Append CCC on test set
        subj_ccc.append(test_ccc)

    # Print CCC values on last epoch for each subject
    print("===")
    print("Subj CCC ")
    for s, ccc in zip(subjects, subj_ccc):
        print("{:4} {:0.3f}".format(s, ccc))
    print("---")
    # Print average personalized CCC
    avg_ccc = sum(subj_ccc) / len(subjects)
    print("Avg. CCC: {:0.3f}".format(avg_ccc))
    print("===")

        
