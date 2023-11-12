## ChebNet
### Paper
- Convolutional neural networks on graphs with fast localized spectral filtering (NeurIPS 2016)

### Methods
- Chebyshev polynomial
$$T_0(x)=1, T_1(x)=x, T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$$
- Chebyshev convolution
$$H^{(l+1)} = \sum_{k=0}^{K} T_k(\tilde{L})H^{(l)}W_{k}^{(l)}$$
Here, $\tilde{L}=2L/ \lambda_{max} - I$, $L = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$ and $\lambda_{max}$ is the largest eigenvalue of matrix $L$.
The hyperparameter $K$ determines the size of the convolution kernel.
