"""Averages predictions in specified folders."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, nargs='+', help='input directories')
    parser.add_argument('--out', type=str, metavar='DIR', default="./averaged",
                        help='output directory (default: ./averaged)')
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    # Extract prediction filenames from first directory
    pattern = "Subject_(\d+)_Story_(\d+)(?:_\w*)?"
    filenames = [fn for fn in sorted(os.listdir(args.dir[0]))
                 if re.match(pattern, fn) is not None]

    # Average data across directories for each filename
    for fn in filenames:
        data = None
        for d in args.dir:
            path = os.path.join(d, fn)
            if data is None:
                data = pd.read_csv(path)
            else:
                data += pd.read_csv(path)
        data /= len(args.dir)
        path = os.path.join(args.out, fn)
        data.to_csv(path, index=False)
