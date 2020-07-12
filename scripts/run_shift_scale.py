
from __future__ import print_function

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--infile", type=str, default="infile")

parser.add_argument("--data_col", type=int, default=-2)
parser.add_argument("--err_col", type=int, default=-1)

parser.add_argument("--max_min_scale_data", type=float, default=1)
parser.add_argument("--shift_last_to_data", type=float, default=0)

parser.add_argument("--max_min_scale_err", type=float, default=1)
parser.add_argument("--shift_last_to_err", type=float, default=0)

args = parser.parse_args()


def scale_min_max(x, factor):
    """
    scale x such that x_max - x_min = factor
    :param x:
    :param factor:
    :return:
    """
    d = x.max() - x.min()
    return (x - x.min()) * factor / d


def shift_last_to(x, target):
    d = target - x[-1]
    return x + d


data = np.loadtxt(args.infile)
header = open(args.infile).readline()

data[:, args.data_col] = scale_min_max(data[:, args.data_col], args.max_min_scale_data)
data[:, args.data_col] = shift_last_to(data[:, args.data_col], args.shift_last_to_data)

data[:, args.err_col] = scale_min_max(data[:, args.err_col], args.max_min_scale_err)
data[:, args.err_col] = shift_last_to(data[:, args.err_col], args.shift_last_to_err)

out_file = os.path.basename(args.infile)
if os.path.exists(out_file):
    raise ValueError(out_file + " exists.")

with open(out_file) as handle:
    handle.write(header)
    for row in data:
        handle.write("%10.5f %10d %10d     %10.5f %10.5f\n" % row)

print("DONE")
