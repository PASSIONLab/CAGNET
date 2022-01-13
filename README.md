# CAGNET: Communication-Avoiding Graph Neural nETworks

## Description

CAGNET is a family of parallel algorithms for training GNNs that can asymptotically reduce communication compared to previous parallel GNN training methods. CAGNET algorithms are based on 1D, 1.5D, 2D, and 3D sparse-dense matrix multiplication, and are implemented with `torch.distributed` on GPU-equipped clusters. We also implement these parallel algorithms on a 2-layer GCN.


For more information, please read our ACM/IEEE SC'20 paper [Reducing Communication in Graph Neural Network Training](https://arxiv.org/pdf/2005.03300.pdf).

**Contact:** Alok Tripathy (<alokt@berkeley.edu>)

## Dependencies
- Python 3.8.11
- PyTorch 1.9.0
- PyTorch Geometric (PyG) 1.7.2
- CUDA 10.2
- GCC 9.3.0

On NERSC Perlmutter, all of these dependencies can be accessed with the following
```bash
module load tudatoolkit/21.3_10.2 # CUDA 10.2
module load pytorch/1.9.0 # PyTorch and PyG
```

On OLCF Summit, older versions of these dependencies can be accessed with the following
```bash
module load cuda # CUDA 10.1
module load gcc # GCC 6.4.0
module load ibm-wml-ce/1.7.0-3 # PyTorch 1.3.1, Python 3.6.10

# PyG and its dependencies
conda create --name gnn --clone ibm-wml-ce-1.7.0-3
conda activate gnn
pip install --no-cache-dir torch-scatter==1.4.0
pip install --no-cache-dir torch-sparse==0.4.3
pip install --no-cache-dir torch-cluster==1.4.5
pip install --no-cache-dir torch-geometric==1.3.2
```

## Compiling

This code uses C++ extensions. To compile these, run

```bash
cd sparse-extension
python setup.py install
```

## Documentation

Each algorithm in CAGNET is implemented in a separate file.
- `examples/gcn_1d.py` : 1D algorithm
- `examples/gcn_15d.py` : 1.5D algorithm
- `examples/gcn_2d.py` : 2D algorithm
- `examples/gcn_3d.py` : 3D algorithm

Each file also as the following flags:

- `--n-epochs <int>`  : Number of epochs to run training
- `--dataset <Cora/Reddit/Amazon/Protein>` : Graph dataset to run training on
- `--timing <True/False>` : Enable timing barriers to time phases in training
- `--n-hidden <int>` : Number of activations in the hidden layer
- `--normalize` : Normalize adjacency matrix in preprocessing
- `--accuracy <True/False>` : Compute and print accuracy metrics (Reddit only)
- `--replication <int>` : Replication factor (1.5D algorithm only)

Some of these flags do not currently exist for the 3D algorithm.

Amazon/Protein datasets must exist as COO files in `../../data/<graphname>/processed/`, compressed with pickle. 
For Cora and Reddit, PyG handles downloading and accessing the dataset.

## Running on NERSC Perlmutter (example)

To run the CAGNET 1.5D algorithm on Reddit with
- 4 processes
- 100 epochs
- 16 hidden layer activations
- 2-factor replication
- normalization
- no weight decay

run the following command with Slurm:

`cd examples && srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 python gcn_15d.py --dataset=Reddit --n-epochs=100 --n-hidden 16 --weight-decay 0 --replication 2 --normalize`

## Citation

To cite CAGNET, please refer to:

> Alok Tripathy, Katherine Yelick, Aydın Buluç. Reducing Communication in Graph Neural Network Training. Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC’20), 2020.
