import torch
import scipy as sp
import scipy.io as spio
from torch_geometric.data import Data

cora_mtx = spio.mmread("/pscratch/sd/a/alokt/graph-vb/build/reddit.mtx")
shuffle_mtx = spio.mmread("../raw/reddit-reordered-k4m1u1c10r2.mtx")

gvb_ids = torch.from_numpy(shuffle_mtx.col).cuda().long()
src_ids = torch.from_numpy(cora_mtx.col).cuda().long()
dataset = torch.load("/pscratch/sd/a/alokt/data/Reddit/processed/data.pt")[0]

src_to_gvb = torch.stack((src_ids, gvb_ids))
src_to_gvb = torch.unique(src_to_gvb, dim=1)

print(src_to_gvb)

src_ids = src_to_gvb[0,:]
gvb_ids = src_to_gvb[1,:]

shuffled_dataset = dict()
print(dataset)

if isinstance(dataset, Data):
    dataset = vars(dataset)["_store"]

print(dataset)

for key in dataset:
    print(key)

    dataset[key] = dataset[key].cuda()
    if key == "edge_index":
        shuffled_dataset[key] = torch.stack((torch.from_numpy(shuffle_mtx.row), \
                                                torch.from_numpy(shuffle_mtx.col))).cuda()
    else:
        shuffled_dataset[key] = torch.empty(dataset[key].size(), dtype=dataset[key][src_ids].dtype).cuda()
        shuffled_dataset[key][gvb_ids] = dataset[key][src_ids].cuda()

for key in dataset:
    if key == "edge_index":
        continue
    print(key)
    print((dataset[key][src_ids] == shuffled_dataset[key][gvb_ids]).all())

print(shuffled_dataset)
torch.save(shuffled_dataset, "data.pt")
