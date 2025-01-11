import torch
from llms.transformer_core.units.attention import MultiHeadAttention as mha
from feedforward import FeedForward as ff
from layer_norm import LayerNorm as ln
from llms.transformer_core.units.embedding import Embedding as embed



class Encoder(torch.nn.Module):
    def __init__(
            self,
            vocabulary_size,
            num_layers=6,
            input_dimension=512,
            output_dimension=512,
            num_heads=8,
            context_length=512,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
    ):
        super().__init__()

        self.num_layers = num_layers
        self.embedding = embed(
            vocabulary_size=vocabulary_size,
            embedding_dimension=input_dimension,
            context_length=context_length
        )
        self.attention = mha(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            num_heads=num_heads,
            context_length=context_length,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.feedforward = ff(
            embedding_dimension=input_dimension,
            scaling_value=scaling_value
        )
        self.norm = ln(
            embedding_dimension=input_dimension,
            epsilon=epsilon
        )

    def forward(self, x):

        x = self.embedding(x)
        for _ in range(self.num_layers):

            attention_residual = x
            x = self.attention(x)
            x = attention_residual + self.norm(x)

            ff_residual = x
            x = self.feedforward(x)
            x = ff_residual + self.norm(x)

        return x




