import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import random

def df_to_graph(df):
    # Define nodes
    x = torch.tensor(df.values, dtype=torch.float)
    # Define edges (connecting each node to its next neighbor)
    edge_index = []
    for i in range(len(x) - 1):
        edge_index.append([i, i + 1])
    edge_index.append([len(x) - 1, 0])      # Connect the last node to the first node
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    return data

def tensor_to_graph(x):
    edge_index = []
    for i in range(len(x) - 1):
        edge_index.append([i, i + 1])
    edge_index.append([len(x) - 1, 0])      # Connect the last node to the first node
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

def df_to_graph_linked(df):
    x = torch.tensor(df.values, dtype=torch.float)
    edge_index = []
    n = len(df) 
    for i in range(n):

        for j in range(i+1, min(n,i+3)):
            edge_index.append([i, j])
            edge_index.append([j, i])

        # for j in range(i+1, i+5):
        #     edge_index.append([i, j%n])
        #     edge_index.append([j%n, i])

        # if min(i,n-i-1) >= 10:
        #     for j in range(i-5, i+6):
        #         edge_index.append([i, j%n])
        #         edge_index.append([j%n, i])
        # else:
        #     for j in range(i-int(min(i,n-i-1)/1.8), i+int(min(i,n-i-1)/1.8)+2):
        #         edge_index.append([i, j%n])
        #         edge_index.append([j%n, i])

        # for j in range(i+1, i+4):
        #     edge_index.append([i, j%n])
        #     edge_index.append([j%n, i])

        # edge_index.append([i, (i - 1)%n])
        # edge_index.append([i, (i - 2)%n])        
        # edge_index.append([i, (i + 1)%n])
        # edge_index.append([i, (i + 2)%n])
        # # Introduce a probabilistic global connection
        # for j in range(n):
        #     distance = min(abs(j - i), n - abs(j - i))
        #     if j != i and distance > 3:
        #         probability = 2.0 / distance**2  # Inverse relationship
        #         if random.random() < probability:
        #             edge_index.append([i, j])

    # edge_index.append([len(x) - 1, 0])      # Connect the last node to the first node
    # edge_index.append([0, len(x) - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    # data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr)
    return data


def edge_indexes(num_graphs, graph_nodes, num_neighbors):
    edge_indexes = []
    for k in range(num_graphs):
        for i in range(k*graph_nodes, (k+1)*graph_nodes):
            for j in range(i+1, i+num_neighbors+1):
                l = j%graph_nodes + k*graph_nodes
                edge_indexes.append([i, l])
                edge_indexes.append([l, i])

    edge_indexes = torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()
    return edge_indexes

def edge_indexes_batch(num_graphs, graph_nodes, num_neighbors):
    edge_index = []
    # edge_indexes = []
    for i in range(graph_nodes):
        for j in range(i+1, i+num_neighbors+1):
            l = j%graph_nodes
            edge_index.append([i, l])
            edge_index.append([l, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # edge_indexes = torch.stack([edge_index] * num_graphs)
    return edge_index