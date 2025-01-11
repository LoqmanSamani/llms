import torch
import math




class FeedForward(torch.nn.Module):
    """
    A FeedForward neural network layer, typically used in Transformer architectures.
    This class implements a position-wise feed-forward network with two linear layers
    and a GELU activation function in between.

    Arguments:
        embedding_dimension (int): The size of the input embedding vector.
        scaling_value (int, optional): The factor by which the hidden layer size is scaled
                                       relative to the embedding dimension. Default is 4.
    """

    def __init__(self, embedding_dimension, scaling_value=4):
        super().__init__()

        # The feed-forward network consists of two linear layers with a GELU activation function in between
        self.layers = torch.nn.Sequential(
            # First Linear Layer: Expands the embedding dimension by a factor of `scaling_value`
            torch.nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * scaling_value,
                bias=True  # Bias is included for better learning capabilities
            ),
            GeluActivation(),  # GELU activation function applied after the first linear layer. torch.nn.GELU can be used instead of GeluActivation
            # Second Linear Layer: Projects the result back to the original embedding dimension
            torch.nn.Linear(
                in_features=embedding_dimension * scaling_value,
                out_features=embedding_dimension,
                bias=True  # Bias is included for better learning capabilities
            )
        )

    def forward(self, x):
        """
        Forward pass through the FeedForward network.

        Arguments:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dimension)
                    after applying the feed-forward network.
        """
        return self.layers(x)


class GeluActivation(torch.nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function. This is an approximation of the
    error function that has been found to work well in Transformer-based architectures.

    The GELU function is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply the GELU activation function to the input tensor.

        Arguments:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor: Output tensor of the same shape with the GELU activation applied element-wise.
        """
        # Applying the GELU activation function using the formula
        a = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return a



# Example input
batch_size = 2
sequence_length = 5
embedding_dimension = 512  # Embedding dimension
scaling_value = 4  # Scaling value for the hidden layer size

# Create some random input tensor of shape (batch_size, sequence_length, embedding_dimension)
input_tensor = torch.randn(batch_size, sequence_length, embedding_dimension)

# Instantiate the FeedForward model with the embedding dimension and scaling value
feedforward = FeedForward(embedding_dimension, scaling_value=scaling_value)

# Forward pass through the FeedForward network
output_tensor = feedforward(input_tensor)

# Print the shape of the input and output tensors to check the transformation
print(f"Input tensor shape: {input_tensor.shape}")  # Expected: (2, 5, 512)
print(f"Output tensor shape: {output_tensor.shape}")  # Expected: (2, 5, 512)

"""
Input tensor shape: torch.Size([2, 5, 512])
Output tensor shape: torch.Size([2, 5, 512])
"""