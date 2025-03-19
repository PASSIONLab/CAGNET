import argparse
import numpy as np
import scipy as sp
import scipy.io as spio
import os

import torch
from torch_geometric.data import InMemoryDataset, Data

def mtx_to_npz(mtx_file):
    filename = mtx_file

    print("Reading mtx", flush=True)
    mtx = spio.mmread(filename)
    mtx = np.stack((mtx.row, mtx.col)).transpose()
    print(f"mtx: {mtx}")
    print(f"type(mtx): {type(mtx)}")
    print("Done reading mtx", flush=True)

    print("Saving .npz", flush=True)
    np.savez_compressed(f"{mtx_file}.npz", a=mtx)
    print("Done saving .npz", flush=True)
    

def npz_to_pickle(mtx_file):
    processed_path = "../processed/"
    filename = f"{mtx_file}.npz"
    
    print("Loading npz", flush=True)
    adj = np.load(os.path.join(".", filename), allow_pickle=True)
    print("Done loading npz", flush=True)

    print(f"adj[a]: {adj['a']}")
    print(f"type(adj[a]): {type(adj['a'])}")
    print(f"adj[a].shape: {adj['a'].shape}")
    edge_index = torch.from_numpy(adj['a'])

    print("Saving data", flush=True)
    torch.save(edge_index, processed_path + f"{mtx_file}.pt")
    print("Done saving data", flush=True)


def main(mtx_file):
    mtx_to_npz(mtx_file)
    npz_to_pickle(mtx_file)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtx-file", type=str, help="mtx file to convert")
    args = parser.parse_args()
    main(args.mtx_file)
