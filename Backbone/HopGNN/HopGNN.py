import torch
from torch import nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from einops import rearrange, repeat

# D-1/2 * A * D-1/2
def get_normalized_adjacent_matrix(edge_index, N):
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :], sparse_sizes=(N, N))
    assert isinstance(adj, SparseTensor)
    size = adj.size(0)
    ones = torch.ones(size).view(-1, 1).to(adj.device())
    degree = adj @ ones
    degree = degree ** (-1 / 2)
    degree[torch.isinf(degree)] = 0
    d = SparseTensor(row=torch.arange(size), col=torch.arange(size), value=degree.squeeze().cpu(),
                     sparse_sizes=(size, size)).to(adj.device())
    adj = adj @ d
    adj = adj * degree
    return adj


class MHSA(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


# A simple version of HopGNN
class HopGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_hop=6, inter_layer=2):
        super().__init__()
        self.num_hop = num_hop
        self.inter_layer = inter_layer

        # hop-embedding
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.E_order = nn.Parameter(torch.randn(1, num_hop, hidden_channels))
        self.embedding_LN = nn.LayerNorm(hidden_channels)

        # hop-interaction
        self.interaction_layers = nn.ModuleList()
        self.interaction_LN = nn.ModuleList()
        for i in range(inter_layer):
            self.interaction_layers.append(MHSA(hidden_channels, heads=4, dropout=0.1))
            self.interaction_LN.append(nn.LayerNorm(hidden_channels))

        # hop-fusion
        self.fusion_LN = nn.LayerNorm(hidden_channels)

        # prediction
        self.prediction = nn.Linear(hidden_channels, out_channels)

    # step 1, build_hop
    def build_hop(self, x, edge_index):
        A_hat = get_normalized_adjacent_matrix(edge_index, x.shape[-2])

        x_hat = []
        x_hat.append(x)
        for i in range(self.num_hop-1):
            x = A_hat @ x
            x_hat.append(x)
        self.x_hat = torch.stack(x_hat, dim=1)
        return self.x_hat

    # step 2, embedding
    def embedding(self, x_hat):
        h_0 = self.fc(x_hat)
        h_0 = h_0 + self.E_order
        h_0 = self.embedding_LN(h_0)
        return h_0

    # step 3, interaction
    def interaction(self, h_k):
        for i in range(len(self.interaction_layers)):
            h_k_pre = h_k
            h_k = self.interaction_layers[i](h_k_pre)
            h_k = F.relu(h_k)
            h_k = h_k + h_k_pre
            h_k = self.interaction_LN[i](h_k)
        return h_k

    # step 4, fusion
    def fusion(self, h_k):
        z = self.fusion_LN(h_k.mean(dim=1))
        return z

    def forward(self, x, adj):
        # step 1, build_hop
        x_hat = self.build_hop(x, adj)
        # step 2, hop-embedding
        h_0 = self.embedding(x_hat)
        # step 3, hop-interaction
        h_k = self.interaction(h_0)
        # step-4 hop-fusion
        z = self.fusion(h_k)
        # step-5 hop-prediction
        y_hat = self.prediction(z)
        return y_hat


if __name__ == '__main__':
    x = torch.randn(100, 64).cuda()
    adj = torch.Tensor([[1,2],[3,4],[5,6],[7,8]]).long().T.cuda()
    model = HopGNN(in_channels=64, hidden_channels=128, out_channels=7).cuda()
    y_hat = model(x, adj)
    print(y_hat.shape)