import torch
from encoder import Encoder

def test_encoder():

    batch_size = 4
    seq_len = 10
    vocab_size = 100
    embed_dim = 512
    output_dim = 512
    num_layers = 6
    num_heads = 8
    context_len = 50
    dropout_rate = 0.1
    scaling_value = 4
    epsilon = 1e-5

    encoder = Encoder(
        vocabulary_size=vocab_size,
        num_layers=num_layers,
        input_dimension=embed_dim,
        output_dimension=output_dim,
        num_heads=num_heads,
        context_length=context_len,
        dropout_rate=dropout_rate,
        qkv_bias=False,
        scaling_value=scaling_value,
        epsilon=epsilon
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = encoder(token_ids)

    assert output.shape == (batch_size, seq_len, output_dim), "Output shape mismatch!"
    assert output.requires_grad, "Output should retain gradients for backpropagation!"
    print("Encoder test passed successfully!")


test_encoder()