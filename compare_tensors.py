# Compares two tensors to see if they have similar values

import argparse
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--tensora", type=str)
parser.add_argument("--tensorb", type=str)
parser.add_argument("--bound", type=float)

args = parser.parse_args()
print(args)

tensora = torch.load(args.tensora)
tensorb = torch.load(args.tensorb)

diff = torch.abs(tensora - tensorb)
diff = torch.lt(diff, args.bound)
print(torch.all(diff))
