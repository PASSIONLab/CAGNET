import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, GraphConv, BatchNorm
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import aggr

class GNNEdgeClassifier(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, n_layers):
        super(GNNEdgeClassifier, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GraphConv(node_input_dim, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        for i in range(n_layers):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(len(self.convs)):
            x = F.gelu(self.convs[i](x, edge_index, edge_weight=edge_attr))
            x = self.norms[i](x)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)

        return torch.sigmoid(self.edge_mlp(edge_features)).squeeze()


def main(args):

    # Load data
    data = np.load('/global/cfs/cdirs/m1982/gnn_binning/training_data.npz')
    print("done loading data")

    # Normalize sequence_lens
    sequence_lens_normalized = (data['sequence_lens'] - 1500) / (5637352 -1500)
    print("normalized sequences")

    # Expand dimensions of sequence_lens_normalized to concatenate with markers
    sequence_lens_normalized = np.expand_dims(sequence_lens_normalized, axis=1)

    # Combine markers and normalized sequence_lens
    X_node_features =  np.concatenate([data['markers'], sequence_lens_normalized], axis=1)
    X_edge_features = data['distances'][:,:75].reshape(-1, 1)
    N, K = data['indices'][:,:75].shape  # N: number of nodes, K: number of neighbors
    print("reshaped")
    m = data['indices'][:,:75]
    edges = (np.array([[i, j] for i in range(N) for j in m[i]]))
    print("h4")

    gl = data['genome_labels']
    edge_labels = np.array([gl[i] == gl[j] for i, j in edges]).astype(int)

    edges = np.transpose(edges)
    print('genome labels')


    n_nodes = len(data['sequence_lens'])
    n_node_features = 41
    n_edge_features = 1

    node_input_dim = 41  # Example node feature dimension
    edge_input_dim = 1  # Example edge feature dimension
    hidden_dim = args.n_hidden  # Hidden dimension size
    model = GNNEdgeClassifier(node_input_dim, edge_input_dim, hidden_dim, args.n_layers)

    # Example data (your data should be formatted similarly)
    num_nodes = n_nodes  # Example number of nodes
    num_edges = n_nodes*75  # Example number of edges
    data = Data(
        x=X_node_features,  # Node features
        edge_index=edges,  # Edge connections
        edge_attr=X_edge_features,
    )

    # Assuming binary classification, you'll need a target for each edge.
    # target = edge_labels.astype(np.float32)  # Example binary targets
    
    # Move data to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    data.x = torch.tensor(data.x, dtype=torch.float32) if not isinstance(data.x, torch.Tensor) else data.x
    data.edge_index = torch.tensor(data.edge_index, dtype=torch.long) \
                            if not isinstance(data.edge_index, torch.Tensor) \
                            else data.edge_index
    data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.float32) \
                            if not isinstance(data.edge_attr, torch.Tensor) \
                                else data.edge_attr
    target = torch.from_numpy(edge_labels).float().to(device)

    model.to(device)
    data = data.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)


    print("percent true",sum(edge_labels)/len(edge_labels))

    # model_path = 'model_weights.pth'  # Path to your saved model weights
    # model.load_state_dict(torch.load(model_path))

    # Training loop
    model.train()
    for epoch in range(args.n_epochs):  # Example epoch count
        optimizer.zero_grad()
        out = model(data)  # This outputs probabilities
        # Convert probabilities to binary predictions
        # Assuming a threshold of 0.5 for binary classification
        predictions = (out > 0.5).float()  # Convert probabilities to 0 or 1
        # Calculate loss (as before)
        loss = F.binary_cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        correct = (predictions == target).float().sum()  # Count correct predictions
        accuracy = correct / target.shape[0]  # Calculate accuracy
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}')

        # model_path = 'model_weights.pth'  # Define the path to save the model weights
        # torch.save(model.state_dict(), model_path)
        # model_path = 'model_weights.pth'  # Path to your saved model weights
        # model.load_state_dict(torch.load(model_path))


    model.eval()  # Set the model to evaluation mode

    # Disable gradient computation for inference
    with torch.no_grad():
        out = model(data)

    # Assuming binary classification and `out` contains the sigmoid output of the model
    predictions = out.squeeze()

    # Convert predictions to a numpy array
    predictions_np = predictions.cpu().numpy()

    # True labels for your training data
    true_labels = target.cpu().numpy()  # Assuming `target` contains your true labels

    # Separate predictions based on their true labels
    positive_predictions = predictions_np[true_labels == 1]
    negative_predictions = predictions_np[true_labels == 0]

    # import matplotlib.pyplot as plt
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.hist(positive_predictions, bins=50, alpha=0.5, density=True, label='Positive Examples')
    # plt.hist(negative_predictions, bins=50, alpha=0.5, density=True,label='Negative Examples')
    # plt.xlabel('Prediction Score')
    # plt.ylabel('Count')
    # plt.title('Distribution of Predictions')
    # plt.legend()
    # plt.ylim([0,6])
    # plt.savefig("/pscratch/sd/r/richardl/condpack/fig_airways2.png")

    from sklearn.metrics import roc_auc_score
    predictions_np = predictions.cpu().detach().numpy() if torch.is_tensor(predictions) else predictions
    targets_np = target.cpu().numpy() if torch.is_tensor(target) else target

    # Calculate AUC
    auc_score = roc_auc_score(targets_np, predictions_np)

    print(f"AUC: {auc_score:.4f}")

    # Calculate AUC using distances
    auc_score = roc_auc_score(targets_np, -X_edge_features)

    print(f"AUC Raw Distances: {auc_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    args = parser.parse_args()
    main(args)
