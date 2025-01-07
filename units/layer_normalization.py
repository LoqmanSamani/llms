import torch


class LayerNorm(torch.nn.Module):
    """
    Layer Normalization (LayerNorm) for normalizing inputs across the features dimension.

    This implementation normalizes the input tensor `x` along the last dimension (features)
    by subtracting the mean and dividing by the standard deviation. It also applies a learnable
    scaling and shifting transformation (gamma and beta) to the normalized tensor.

    Args:
        embedding_dimension (int): The size of the feature dimension (e.g., number of features in the input).
        epsilon (float, optional): A small constant added to the variance for numerical stability (default: 1e-5).

    Attributes:
        scale (torch.nn.Parameter): Learnable parameter that scales the normalized output.
        shift (torch.nn.Parameter): Learnable parameter that shifts the normalized output.
    """

    def __init__(self, embedding_dimension, epsilon=1e-5):
        """
        Initializes the LayerNorm layer with learnable scaling and shifting parameters.

        Args:
            embedding_dimension (int): The size of the feature dimension to normalize over.
            epsilon (float): Small value to avoid division by zero in variance computation.
        """
        super().__init__()

        # Initialize epsilon, scale (gamma), and shift (beta) parameters
        self.epsilon = epsilon
        self.scale = torch.nn.parameter.Parameter(torch.ones(embedding_dimension))  # Scaling parameter (gamma)
        self.shift = torch.nn.parameter.Parameter(torch.zeros(embedding_dimension))  # Shifting parameter (beta)

    def forward(self, x):
        """
        Forward pass through the LayerNorm layer.

        This function normalizes the input tensor `x` along the last dimension (features),
        and applies the learnable scale and shift transformations.

        Args:
            x (torch.Tensor): The input tensor to normalize. Expected shape:
                              (batch_size, num_tokens, embedding_dimension)

        Returns:
            torch.Tensor: The normalized and scaled tensor with the same shape as the input tensor.
        """
        # Calculate mean and variance along the last dimension (features)
        mean = x.mean(dim=-1, keepdims=True)  # Mean of the features (along last dimension)
        variance = x.var(dim=-1, keepdims=True, unbiased=False)  # Variance of the features (along last dimension)

        # Normalize the input (subtract mean and divide by std deviation)
        x_norm = (x - mean) / torch.sqrt(variance + self.epsilon)  # Normalize with added epsilon for stability

        # Apply learnable scale (gamma) and shift (beta) to the normalized input
        # Scaling: multiplication by scale (gamma)
        # Shifting: adding shift (beta)
        return self.scale * x_norm + self.shift






# Test LayerNorm with a batch of data
batch_size = 2
seq_length = 5
embedding_dimension = 4  # Example embedding dimension
epsilon = 1e-5

# Create an instance of the LayerNorm class
layer_norm = LayerNorm(embedding_dimension, epsilon)

# Create a dummy input tensor with random values
# Shape: (batch_size, seq_length, embedding_dimension)
x = torch.randn(batch_size, seq_length, embedding_dimension)

# Print input tensor
print("Input tensor:")
print(x)
"""
Input tensor:
tensor([[[ 0.2725, -0.1480,  0.2478, -0.3707],
         [ 0.3945,  1.1074,  0.9473,  0.3650],
         [ 0.6822,  1.2001,  1.7417,  1.6067],
         [ 1.4500,  1.7813, -0.9739, -0.1313],
         [-1.3399, -1.2407, -0.2531,  1.2533]],

        [[ 1.3482, -0.6312,  1.2006,  0.9291],
         [-0.1615,  2.1465,  0.4642,  1.1020],
         [-1.5547,  0.2597, -0.0178,  0.8630],
         [-1.0068,  0.0888,  0.0767,  0.5099],
         [ 0.8161,  1.0711,  0.1978,  0.1828]]])

"""


# Apply LayerNorm to the input tensor
output = layer_norm(x)

# Print output tensor after applying LayerNorm
print("\nOutput tensor after LayerNorm:")
print(output)
"""
Output tensor after LayerNorm:
tensor([[[ 1.0019, -0.5466,  0.9111, -1.3663],
         [-0.9397,  1.2280,  0.7411, -1.0293],
         [-1.5163, -0.2608,  1.0522,  0.7248],
         [ 0.8125,  1.1056, -1.3317, -0.5864],
         [-0.9065, -0.8113,  0.1362,  1.5816]],

        [[ 0.8060, -1.7004,  0.6191,  0.2753],
         [-1.2301,  1.4755, -0.4966,  0.2511],
         [-1.6178,  0.4174,  0.1062,  1.0942],
         [-1.6462,  0.3058,  0.2843,  1.0561],
         [ 0.6433,  1.3015, -0.9532, -0.9917]]], grad_fn=<AddBackward0>)
"""


# Check if the normalization is correct (mean ~ 0, std ~ 1)
print("\nMean of output tensor (should be close to 0):")
print(output.mean(dim=-1))  # Mean along the feature dimension
"""
Mean of output tensor (should be close to 0):
tensor([[ 0.0000e+00, -2.9802e-08,  1.4901e-07, -1.4901e-08,  2.9802e-08],
        [ 1.4901e-08,  2.2352e-08,  0.0000e+00,  0.0000e+00, -2.9802e-08]],
       grad_fn=<MeanBackward1>)
"""


print("\nStandard deviation of output tensor (should be close to 1):")
print(output.std(dim=-1))  # Standard deviation along the feature dimension
"""
Standard deviation of output tensor (should be close to 1):
tensor([[1.1546, 1.1546, 1.1547, 1.1547, 1.1547],
        [1.1547, 1.1547, 1.1547, 1.1547, 1.1547]], grad_fn=<StdBackward0>)
"""
