# CAGNET: Communication-Avoiding Graph Neural nETworks

## Description

CAGNET is a family of parallel algorithms for training GNNs that can asymptotically reduce communication compared to previous parallel GNN training methods. CAGNET algorithms are based on 1D, 1.5D, 2D, and 3D sparse-dense matrix multiplication, and are implemented with `torch.distributed` on GPU-equipped clusters. We also implement these parallel algorithms on a 2-layer GCN.

This branch is the artifact code for our ACM ICPP'24 paper [Sparsity-Aware Communication for Distributed Graph Neural Network Training](https://dl.acm.org/doi/pdf/10.1145/3673038.3673152). This paper accelerates full-batch GNN training with sparsity-aware algorithms and [Graph-VB](https://github.com/roguzsel/graph-vb) partitioning.

**Contact:** Alok Tripathy (<alokt@berkeley.edu>)

## Dependencies
- Python 3.9.19
- PyTorch 1.13.0
- PyTorch Geometric (PyG) 2.5.3
- CUDA 11.7
- GCC 11.2

On NERSC Perlmutter, all of these dependencies can be accessed with the following

```bash
module load pytorch/1.13.1
```
## Compiling

This code uses C++/CUDA extensions. To compile these, run

```bash
cd sparse-extension
mkdir build
python setup.py install --prefix build/
export PYTHONPATH="<path_to_CAGNET>/CAGNET/sparse-extension/build/lib/python3.9/site-packages/sparse_coo_tensor_cpp-0.0.0-py3.9-linux-x86_64.egg":$PYTHONPATH
export PYTHONPATH="<path_to_CAGNET>/CAGNET/":$PYTHONPATH
```

## Documentation

Each algorithm in CAGNET is implemented in a separate file.
- `examples/gcn_1d.py` : 1D algorithm
- `examples/gcn_15d.py` : 1.5D algorithm

Each file also as the following flags:

- `--n-epochs <int>`  : Number of epochs to run training
- `--dataset <Cora/Reddit/Amazon/Protein>` : Graph dataset to run training on
- `--timers` : Enable timing barriers to time phases in training
- `--n-hidden <int>` : Number of activations in the hidden layer
- `--normalize` : Normalize adjacency matrix in preprocessing
- `--replication <int>` : Replication factor (1.5D algorithm only)
- `--hostname <str>` : Hostname for node running rank 0 during multi-node training
- `--partitions <path>` : Path to graph-vb output file with partition sizes
- `--sparse-uonaware` : Run sparsity-unaware code

Cora, Amazon, and Protein (subgraph3) mtx files for different partitionings can be accessed on the [NERSC portal](https://portal.nersc.gov/project/m1982/GNN/).
For Cora and Reddit without partitoning, PyG handles downloading and accessing the dataset.

## Running on NERSC Perlmutter (example)

To run the CAGNET 1D algorithm on Cora with
- 16 processes
- 100 epochs
- 16 hidden layer activations
- normalization

run the following command with Slurm:

`cd examples && srun -l -n 16 --cpus-per-task 32 --gpus-per-node 4 python gcn_1d.py --n-epochs=100 --dataset=Cora_16 --n-hidden=16 --normalize --timers --hostname <rank0_hostname> --partitions <path_to_partsizes>/cora-reordered-k16m1u1c10r2.part-sizes`

To run the CAGNET 1D algorithm on Amazon with the above settings,

`cd examples && srun -l -n 16 --cpus-per-task 32 --gpus-per-node 4 python gcn_1d.py --n-epochs=100 --dataset=Amazon_Large_16 --n-hidden=16 --normalize --timers --hostname <rank0_hostname> --partitions <path_to_partsizes>/amazon_large_randomized-reordered-k16m1u0c10r2.part-sizes`

Note that feature data for Amazon is randomly generated, yielding meaningful epoch time performance results but not meaningful accuracy results. Cora and Reddit datasets contain meaningful feature data and converge to their respective accuracies.

## Running other datasets

To train a dataset not included here, users must first 1) partition the graph with [Graph-VB](https://github.com/roguzsel/graph-vb) and 2) convert the output mtx file into a pickled file that PyTorch can load. The Graph-VB repository has instructions for running volume-balancing partitioning. To convert the output mtx file, please run

`python mtx_to_coo.py --mtx-file <path_to_mtx>`

## Citation

To cite our ICPP'24 work, please use:

```
@inproceedings{mukhopadhyay2024sparsity,
  title = {Sparsity-Aware Communication for Distributed Graph Neural Network Training},
  author = {Mukhopadhyay, Ujjaini and Tripathy, Alok and Selvitopi, Oguz and Yelick, Katherine and Bulu{\c{c}}, Ayd{\i}n},
   booktitle = {ACM International Conference on Parallel Processing (ICPP)},
  year = {2024}
}
```
