## GCN
### Paper
- SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (ICLR 2017)

### Framework
<img src="./GCN.png" width="60%">

### Methods
$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}).$$
Here, $\tilde{A} = A + I_N$ is the adjacency matrix of the undirected graph $G$ with added self-connections. $I_N$ is the identity matrix, $\tilde{D}_{ii}=\sum _{j} \tilde{A} _{ij}$ and $W^{(l)}$ is a layer-specific trainable weight matrix. $\sigma(\cdot)$ denotes an activation function, such as the $ReLU(\cdot) = max(0,\cdot)$. $H^{(l)} \in \mathbb{R}^{N \times D}$ is the matrix of activations in the $l^{th}$ layer; $H^{(0)} = X$.
