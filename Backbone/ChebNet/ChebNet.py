import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian


def get_Laplacian_matrix(edge_index, N, lambda_max=None):
    if not lambda_max:
        lambda_max = 2
    edge_index, edge_weight = get_laplacian(edge_index, normalization="sym", num_nodes=N)
    L = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=[N, N], device=edge_index.device)
    I = torch.sparse_coo_tensor(indices=torch.LongTensor([range(N), range(N)]), values=torch.ones(N), size=[N, N],
                                device=edge_index.device)
    L_hat = 2 * L / lambda_max - I
    return L_hat


class chebConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(chebConv, self).__init__()
        self.K = K
        self.Ws = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])

    def forward(self, x, L):
        T_0 = x
        T_1 = L @ x
        for j in range(self.K):
            if j == 0:
                out = T_0 @ self.Ws[j].weight.T
            elif j == 1:
                out = out + T_1 @ self.Ws[j].weight.T
            else:
                T_2 = 2 * L @ T_1 - T_0
                out = out + T_2 @ self.Ws[j].weight.T
                T_0, T_1 = T_1, T_2
        return out


class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, layers=2):
        super(ChebNet, self).__init__()
        self.pre = chebConv(in_channels, hidden_channels, K)
        self.out = chebConv(hidden_channels, out_channels, K)

        self.hiddens = nn.ModuleList([])
        for i in range(layers-2):
            self.hiddens.append(chebConv(hidden_channels, hidden_channels, K))

    def forward(self, x, edge_index):
        L = get_Laplacian_matrix(edge_index, x.shape[0])
        x = F.relu(self.pre(x, L))
        for module in self.hiddens:
            x = x + module(x, L)
            x = F.relu(x)
        x = self.out(x, L)
        return x
