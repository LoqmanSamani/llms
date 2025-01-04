import torch


class Embedding(torch.nn.Module):
    """
    A combined embedding layer for token embeddings and positional encodings.

    This class provides a mechanism to map token IDs to learned embeddings
    and add positional encodings to account for the order of tokens in a sequence.
    Both the token embeddings and positional encodings are learnable parameters.

    Methods:
        positional_encoding(token_ids):
            Computes the positional encodings based on token positions in the sequence.
        token_embedding(token_ids):
            Computes the token embeddings based on token IDs.
        forward(token_ids):
            Combines token embeddings and positional encodings to produce the final embeddings.
    """

    def __init__(self, vocabulary_size, embedding_dimension, context_length):
        """
        Initializes the embedding layer with token and positional embeddings.

        Args:
            vocabulary_size (int): Total number of unique tokens in the vocabulary.
            embedding_dimension (int): Dimensionality of each embedding vector.
            context_length (int): Maximum sequence length supported by positional encodings.
        """
        super().__init__()
        # Embedding layer for token IDs. Maps token indices to learnable embedding vectors.
        self.tok_embed = torch.nn.Embedding(
            num_embeddings=vocabulary_size,  # Vocabulary size (total unique tokens).
            embedding_dim=embedding_dimension  # Size of each embedding vector.
        )

        # Embedding layer for positional encodings. Maps positions to learnable embedding vectors.
        self.pos_encod = torch.nn.Embedding(
            num_embeddings=context_length,  # Maximum sequence length supported.
            embedding_dim=embedding_dimension  # Size of each positional encoding vector.
        )

    def positional_encoding(self, token_ids):
        """
        Computes the positional encodings for a given sequence of token IDs.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (seq_length, embedding_dimension) representing positional encodings.
        """
        # Generate positional indices for the sequence (0, 1, 2, ..., seq_length-1).
        positional_indices = torch.arange(token_ids.shape[-1], device=token_ids.device)

        # Look up the learnable positional encodings for these indices.
        return self.pos_encod(positional_indices)

    def token_embedding(self, token_ids):
        """
        Computes the token embeddings for a given sequence of token IDs.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (batch_size, seq_length, embedding_dimension) representing token embeddings.
        """
        # Look up the learnable token embeddings for the provided token IDs.
        return self.tok_embed(token_ids)

    def forward(self, token_ids):
        """
        Combines token embeddings and positional encodings to produce final embeddings.

        Args:
            token_ids (Tensor): A tensor of shape (batch_size, seq_length) containing token IDs.

        Returns:
            Tensor: A tensor of shape (batch_size, seq_length, embedding_dimension) where each token embedding
                    is enriched with positional encoding.
        """
        # Get token embeddings using the token IDs.
        token_embeds = self.token_embedding(token_ids=token_ids)

        # Get positional encodings for the sequence positions.
        position_embeds = self.positional_encoding(token_ids=token_ids)

        # Combine the token embeddings and positional encodings.
        return token_embeds + position_embeds


# Define parameters
vocabulary_size = 50      # Total number of unique tokens in the vocabulary
embedding_dimension = 50    # Size of each embedding vector
context_length = 50         # Maximum sequence length

# Initialize the embedding layer
embedding_layer = Embedding(vocabulary_size, embedding_dimension, context_length)

# Create sample token IDs (e.g., sentences represented as token indices)
# Batch size: 2, Sequence length: 5
token_ids = torch.tensor([
    [4, 8, 3, 1, 2],
    [4, 9, 7, 5, 3]
])  # Shape: (2, 5)


# Forward pass: Get the combined embeddings
output = embedding_layer(token_ids)

# Print the shapes of inputs and outputs
print("Token IDs shape:", token_ids.shape)        # (batch_size, seq_length)
print("Output embeddings shape:", output.shape)  # (batch_size, seq_length, embedding_dimension)

"""
Token IDs shape: torch.Size([2, 5])
Output embeddings shape: torch.Size([2, 5, 10])
"""

# Output the embeddings for inspection
print("Combined embeddings:", output)

"""
Combined embeddings:
 tensor([[[-0.5938, -0.9668,  1.9755, -1.7791,  0.5572, -1.6489, -0.9753,
           0.0368,  1.0664, -0.0505],
         [-2.7086, -1.8036, -1.2247,  1.9863, -0.0579, -0.2869, -2.4279,
          -0.2722, -2.4865,  1.5684],
         [ 1.1199,  1.7105,  0.3264, -1.8995,  0.4361,  1.3117,  0.1379,
           2.2836, -1.5669,  2.4284],
         [-3.0090, -0.1908,  0.8124, -1.7515, -0.6078, -0.0712, -0.8220,
          -1.2441,  0.4427,  0.4786],
         [ 1.4154,  0.1694,  1.4462,  1.3556,  0.4407,  1.1827,  1.2919,
           0.8737,  0.0689, -0.2870]],

        [[-0.5938, -0.9668,  1.9755, -1.7791,  0.5572, -1.6489, -0.9753,
           0.0368,  1.0664, -0.0505],
         [-2.2743, -2.0574,  1.0986,  0.4097,  0.6563, -1.1021, -0.7460,
           0.4668, -2.1977, -0.9089],
         [-0.8947, -0.8457, -0.3148, -0.6253,  1.3851,  2.3509, -0.7972,
           1.7073,  0.0103,  1.6783],
         [-2.6242, -0.5367,  0.6617, -1.2934, -1.9505,  1.3493, -0.4510,
          -1.1136,  1.3015, -0.3611],
         [ 1.8978,  1.3817,  1.0710, -0.8778,  0.4170, -0.3248, -0.7044,
           0.1070, -0.0840,  1.6013]]], grad_fn=<AddBackward0>)
"""

print(output.shape)
"""torch.Size([2, 5, 10])"""