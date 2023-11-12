import torch
import torch.nn as nn
import torch.nn.functional as F


def degree(edge_index, N):
    edge_index = edge_index[0,:]
    out = torch.zeros((N, ), device=edge_index.device)
    one = torch.ones((edge_index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, edge_index, one)


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        N, C = x.shape
        deg = degree(edge_index, N)
        A = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1]), size=[N, N],
                                    device=x.device)
        D = torch.sparse_coo_tensor(indices=torch.LongTensor([range(N), range(N)]), values=deg, size=[N, N],
                                    device=x.device)
        I = torch.sparse_coo_tensor(indices=torch.LongTensor([range(N), range(N)]), values=torch.ones(N), size=[N, N],
                                    device=x.device)

        A_tilde = A + I
        D_tilde = D + I
        x = torch.matmul(D_tilde ** -0.5, x)
        x = torch.matmul(A_tilde, x)
        x = torch.matmul(D_tilde ** -0.5, x)
        x = torch.matmul(x, self.W.weight.T)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers=2):
        super(GCN, self).__init__()
        self.pre = GCNConv(in_channels, hidden_channels)
        self.out = GCNConv(hidden_channels, out_channels)

        self.hiddens = nn.ModuleList([])
        for i in range(layers-2):
            self.hiddens.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        x = F.relu(self.pre(x, edge_index))
        for module in self.hiddens:
            x = x + module(x, edge_index)
            x = F.relu(x)
        x = self.out(x, edge_index)
        return x
