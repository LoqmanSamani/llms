import torch



class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention mechanism as described in the Transformer architecture.

    This module implements the multi-head attention mechanism, which allows
    the model to jointly attend to information from different representation
    subspaces at different positions. It performs the scaled dot-product attention
    mechanism in parallel across multiple attention heads, followed by a linear projection.

    Args:
        input_dimension (int): The number of input features (dimensionality of input vectors).
        output_dimension (int): The number of output features (dimensionality of the attention output).
        num_heads (int, optional): The number of attention heads. Default is 8.
        context_length (int, optional): The maximum length of the input sequence (for mask generation). Default is 512.
        dropout_rate (float, optional): The dropout probability to apply on attention weights. Default is 0.1.
        qkv_bias (bool, optional): If True, add a bias term to the Q, K, V linear projections. Default is False.
    """

    def __init__(self, input_dimension=512, output_dimension=512, num_heads=8, context_length=512, dropout_rate=0.1, qkv_bias=False):
        super().__init__()

        # Ensure output_dimension is divisible by num_heads
        assert output_dimension % num_heads == 0, "Output dimension must be divisible by the number of heads"

        # Initialize hyperparameters
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        self.head_dimension = self.output_dimension // self.num_heads

        # Define the linear projections for queries, keys, and values
        self.query_params = torch.nn.Linear(
            in_features=self.input_dimension,
            out_features=self.output_dimension,
            bias=qkv_bias
        )
        self.key_params = torch.nn.Linear(
            in_features=self.input_dimension,
            out_features=self.output_dimension,
            bias=qkv_bias
        )
        self.value_params = torch.nn.Linear(
            in_features=self.input_dimension,
            out_features=self.output_dimension,
            bias=qkv_bias
        )

        # Linear projection for the output of the attention mechanism
        self.out_projection = torch.nn.Linear(
            in_features=self.output_dimension,
            out_features=self.output_dimension
        )

        # Dropout layer for attention weights
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Mask to prevent attending to future tokens (for autoregressive models)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length,dtype=torch.bool),diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_tokens, input_dimension).

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention, shape (batch_size, num_tokens, output_dimension).
        """
        # Get the batch size, number of tokens, and input dimension from the input tensor
        batch_size, num_tokens, input_dimension = x.shape

        # Ensure the input dimension matches the expected input dimension
        assert input_dimension == self.input_dimension, "Input dimension mismatch"

        # Compute the query, key, and value projections
        queries = self.query_params(x)  # (batch_size, num_tokens, output_dimension)
        keys = self.key_params(x)  # (batch_size, num_tokens, output_dimension)
        values = self.value_params(x)  # (batch_size, num_tokens, output_dimension)

        # Reshape and transpose the projections to separate heads
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dimension).transpose(1, 2)

        # Compute the scaled dot-product attention scores
        attn_scores = queries @ keys.transpose(-2, -1) / (
                    self.head_dimension ** 0.5)  # (batch_size, num_heads, num_tokens, num_tokens)

        # Apply the mask to prevent attending to future tokens (for autoregressive tasks)
        mask = self.mask[:num_tokens, :num_tokens]  # Adjust the mask size based on num_tokens
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))  # Apply the mask

        # Apply softmax to get the attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Normalize attention scores
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        # Compute the attention output (context vector)
        context_vector = attn_weights @ values  # (batch_size, num_heads, num_tokens, head_dimension)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.output_dimension)

        # Apply the output projection
        output_ = self.out_projection(context_vector)  # (batch_size, num_tokens, output_dimension)

        return output_





batch_size = 2
num_tokens = 10
input_dim = 32
output_dim = 64
num_heads = 8
context_length = 512

x = torch.rand(batch_size, num_tokens, input_dim)
attention_layer = MultiHeadAttention(input_dim, output_dim, num_heads, context_length)
output = attention_layer(x)
print(output.shape)  # Expected: (batch_size, num_tokens, output_dim)

"""
torch.Size([2, 10, 64])
tensor([[[ 0.0223, -0.2179, -0.3280,  ...,  0.1542, -0.2348, -0.1023],
         [ 0.1726, -0.1985, -0.2771,  ...,  0.0795, -0.1851, -0.1776],
         [ 0.1849, -0.1811, -0.2682,  ...,  0.0941, -0.2171, -0.2274],
         ...,
         [ 0.1799, -0.1247, -0.2404,  ...,  0.0876, -0.0222, -0.1397],
         [ 0.2064, -0.1073, -0.2442,  ...,  0.0428, -0.0707, -0.2109],
         [ 0.2133, -0.1345, -0.2666,  ...,  0.0272, -0.0870, -0.1921]],

        [[-0.0007, -0.0921, -0.3059,  ...,  0.0388,  0.0720, -0.0381],
         [-0.0180, -0.1374, -0.2019,  ...,  0.0860, -0.0123, -0.0501],
         [ 0.1131, -0.0423, -0.2825,  ..., -0.0105, -0.0913, -0.1769],
         ...,
         [ 0.2060, -0.1149, -0.2928,  ...,  0.0669, -0.0733, -0.2287],
         [ 0.1935, -0.1804, -0.2499,  ...,  0.0688, -0.1024, -0.2126],
         [ 0.2071, -0.1705, -0.2698,  ...,  0.0390, -0.0929, -0.2317]]],
       grad_fn=<ViewBackward0>)
"""

