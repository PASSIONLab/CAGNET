# CAGNET: Communication-Avoiding Graph Neural nETworks

## Description

CAGNET is a family of parallel algorithms for training GNNs that can asymptotically reduce communication compared to previous parallel GNN training methods. CAGNET algorithms are based on 1D, 1.5D, 2D, and 3D sparse-dense matrix multiplication, and sparse-sparse matrix multiplication. All algorithms are implemented with `torch.distributed` on GPU-equipped clusters.


This branch contains a minibatch GNN training pipeline that implements GraphSAGE and LADIES sampling using sparse-sparse matrix multiplication. For more information, please read our MLSys'24 paper [Distributed Matrix-Based Sampling for Graph Neural Network Training](https://arxiv.org/abs/2311.02909).

**Contact:** Alok Tripathy (<alokt@berkeley.edu>)

## Dependencies
- Python 3.9.19
- PyTorch 1.13
- PyTorch Geometric (PyG) 2.5.3
- CUDA 11.7
- GCC 11.2

On NERSC Perlmutter, you can run the following to load PyTorch and PyG.
```bash
module load pytorch/1.13 

```
## Compiling

This code uses C++ extensions. To compile these, run

```bash
cd sparse-extension
mkdir build
python setup.py install --prefix build
export PYTHONPATH="<parent_dir>/CAGNET/sparse-extension/build/lib/python3.9/site-packages/sparse_coo_tensor_cpp-0.0.0-py3.9-linux-x86_64.egg:/":$PYTHONPATH
```

## Documentation

Our minibatch training pipeline, including both our SAGE and LADIES implementations, can be accessed by running
- `examples/gcn_15d.py` : 1.5D algorithm

This file also has the following flags:

- `--n-epochs <int>`  : Number of epochs to run training
- `--dataset <cora/reddit/Amazon/Protein/ogbn-products/ogbn-papers100M>` : Graph dataset to run training on
- `--timing` : Enable timing barriers to time phases in training
- `--n-hidden <int>` : Number of activations in the hidden layer
- `--replication <int>` : Replication factor 
- `--sample-method <sage/ladies>` : Sampling algorithm to use
- `--n-layers <int>` : Number of hidden gnn layers
- `--batch-size <int>` : Number of vertices in a minibatch
- `--samp-num <int-int-int..>` : Dash-separated list of sample number (i.e. number of neighbors to sample for SAGE, or number of vertices in the next layer for LADIES).
- `--n-bulkmb <int>` : Number of minibatches to sample in bulk during one step of training.
- `--semibulk <int>` : Number of minibatches to column extract in bulkb during one LADIES call.
- `--replicate-graph` : Replicate the input graph on each GPU.

## Running on NERSC Perlmutter (example)

To run the CAGNET 1.5D algorithm on ogbn-products with
- 4 GPUs
- 30 epochs
- GraphSAGE sampling
- 1024 batch size
- 15, 10, 5 sample numbers
- 3 layers
- 48 minibatches bulk sampled per step
- 256 hidden layer activations
- Replication factor 1 for feature data
- Graph replication

run the following command with Slurm:

`cd examples && srun -l -n 4 --cpus-per-task 32 --ntasks-per-node 4 --gpus-per-node 4 python gcn_15d.py --dataset ogbn-products --sample-method sage --n-epochs 30 --batch-size 1024 --n-layers 3 --samp-num 15-10-5 --replication 1 --n-bulkmb 48 --replicate-graph`

## Citation

To cite CAGNET, please refer to:

> Alok Tripathy, Katherine Yelick, Aydın Buluç. Distributed Matrix-Based Sampling for Graph Neural Network Training. Proceedings of Machine Learning and Systems (MLSys’24), 2024.
