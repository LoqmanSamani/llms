import torch
from feed_forward import FeedForward
from self_attention import MultiHeadAttention
from layer_normalization import LayerNorm


class Transformer(torch.nn.Module):
    def __init__(
            self,
            input_dimension=512,
            output_dimension=512,
            num_heads=8,
            context_length=512,
            dropout_rate=0.1,
            qkv_bias=False,
            layer_norm_epsilon=1e-5,
            ff_scaling_value=4
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.layer_norm1 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.layer_norm2 = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.feed_forward = FeedForward(
            embedding_dimension=input_dimension,
            scaling_value=ff_scaling_value
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        

    def forward(self, x):

        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x



# Hyperparameters
batch_size = 2  # Number of sequences in a batch
context_length = 512  # Length of each sequence
input_dimension = 512  # Embedding dimension

# Instantiate the Transformer
model = Transformer(
    input_dimension=input_dimension,
    output_dimension=input_dimension,  # Typically the same as input_dimension
    num_heads=8,
    context_length=context_length,
    dropout_rate=0.1,
    qkv_bias=True,
    layer_norm_epsilon=1e-5,
    ff_scaling_value=4
)

# Create dummy input tensor
# Shape: (batch_size, context_length, input_dimension)
dummy_input = torch.randn(batch_size, context_length, input_dimension)

# Run the input through the model
output = model(dummy_input)

# Print output shape to confirm it's correct
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

"""
Input shape: torch.Size([2, 512, 512])
Output shape: torch.Size([2, 512, 512])
"""
