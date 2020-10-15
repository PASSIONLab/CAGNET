# CAGNET: Communication-Avoiding Graph Neural nETworks

## Description

CAGNET is a family of parallel algorithms for training GNNs that can asymptotically reduce communication compared to previous parallel GNN training methods. CAGNET algorithms are based on 1D, 1.5D, 2D, and 3D sparse-dense matrix multiplication, and are implemented with `torch.distributed` on GPU-equipped clusters. We also implement these parallel algorithms on a 2-layer GCN.


For more information, please read our ACM/IEEE SC'20 paper [Reducing Communication in Graph Neural Network Training](https://arxiv.org/pdf/2005.03300.pdf).

**Contact:** Alok Tripathy (<alokt@berkeley.edu>)

## Dependencies
- Python 3.6.10
- PyTorch 1.3.1
- PyTorch Geometric (PyG) 1.3.2
- CUDA 10.1
- GCC 6.4.0

On OLCF Summit, all of these dependencies can be accessed with the following
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
- `gcn_distr.py` : 1D algorithm
- `gcn_distr_15d.py` : 1.5D algorithm
- `gcn_distr_2d.py` : 2D algorithm
- `gcn_distr_3d.py` : 3D algorithm

Each file also as the following flags:

- `--accperrank <int>` : Number of GPUs on each node
- `--epochs <int>`  : Number of epochs to run training
- `--graphname <Reddit/Amazon/subgraph3>` : Graph dataset to run training on
- `--timing <True/False>` : Enable timing barriers to time phases in training
- `--midlayer <int>` : Number of activations in the hidden layer
- `--runcount <int>` : Number of times to run training
- `--normalization <True/False>` : Normalize adjacency matrix in preprocessing
- `--activations <True/False>` : Enable activation functions between layers
- `--accuracy <True/False>` : Compute and print accuracy metrics (Reddit only)
- `--replication <int>` : Replication factor (1.5D algorithm only)
- `--download <True/False>` : Download the Reddit dataset

Some of these flags do not currently exist for the 3D algorithm.

## Running on OLCF Summit (example)

To run the CAGNET 1.5D algorithm on Reddit with
- 16 processes
- 100 epochs
- 16 hidden layer activations
- 2-factor replication

run the following command to download the Reddit dataset:

`python gcn_distr_15d.py --graphname=Reddit --download=True`

This will download Reddit into `../data`. After downloading the Reddit dataset, run the following command to run training

`ddlrun -x WORLD_SIZE=16 -x MASTER_ADDR=$(echo $LSB_MCPU_HOSTS | cut -d " " -f 3) -x MASTER_PORT=1234 -accelerators 6 python gcn_distr_15d.py --accperrank=6 --epochs=100 --graphname=Reddit --timing=False --midlayer=16 --runcount=1 --replication=2`

## Citation

To cite CAGNET, please refer to:

> Alok Tripathy, Katherine Yelick, Aydın Buluç. Reducing Communication in Graph Neural Network Training. Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC’20), 2020.
