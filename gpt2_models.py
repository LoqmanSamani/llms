import torch
import tiktoken



class GPT2Model(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, context_length, dropout_rate, num_layer, num_heads, qkv_bias=False):
        super().__init__()
        # num_embeddings(int) – size of the dictionary of embeddings
        # embedding_dim(int) – the size of each embedding vector
        self.token_embed = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        ) # token embedding layer
        self.position_embed = torch.nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embed_dim
        ) # positional embedding layer
        self.dropout = torch.nn.Dropout(p=dropout_rate) # dropout layer
        self.transformer_blocks = torch.nn.Sequential(
            *[Transformer(
                num_heads=num_heads,
                embed_dim=embed_dim,
                context_length=context_length,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias
            )
                for _ in range(num_layer)]
        )
        self.final_norm = LayerNorm(embed_dim=embed_dim)
        self.linear_transform = torch.nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            bias=False
        )

    def forward(self, in_idx):

        batch_size, seq_length = in_idx.shape
        token_embeds = self.token_embed(in_idx)
        position_embeds = self.position_embed(
                    torch.arange(seq_length, device=in_idx.device)
                )
        x = token_embeds + position_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.linear_transform(x)

        return logits




class Transformer(torch.nn.Module):

    def __init__(self, num_heads, embed_dim, context_length, dropout_rate, qkv_bias):
        super().__init__()
        self.attention = MultiHeadAttention(
            in_dim=embed_dim,
            out_dim=embed_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias
        )
        self.feed_forward = FeedForward(embed_dim=embed_dim)
        self.norm1 = LayerNorm(embed_dim=embed_dim)
        self.norm2 = LayerNorm(embed_dim=embed_dim)
        self.drop_shortcut = torch.nn.Dropout(dropout_rate)


    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class LayerNorm(torch.nn.Module):

    def __init__(self, embed_dim, epsilon=1e-5):
        super().__init__()

        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones(embed_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)

        return self.scale * norm_x + self.shift


class FeedForward(torch.nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            GELU(),
            torch.nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.layers(x)


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))



class MultiHeadAttention(torch.nn.Module):

    def __init__(self, in_dim, out_dim, context_length, dropout_rate, num_heads, qkv_bias):
        super().__init__()
        assert (out_dim % num_heads == 0) # make sure the out_dim is divisible by the number of heads
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W_query = torch.nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.out_projection = torch.nn.Linear(in_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # Masks truncated to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # Uses the mask to fill attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.out_dim)
        context_vector = self.out_projection(context_vector)  # Adds an optional linear projection

        return context_vector


GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size (a vocabulary of 50,257 words, as used by the BPE tokenizer)
    "context_length": 1024, # Context length ( the maximum number of input tokens the model can handle via the positional embeddings)
    "emb_dim": 768, # Embedding dimension (embedding size, transforming each token into a 768-dimensional vector)
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers (number of transformers blocks)
    "drop_rate": 0.1, # Dropout rate (0.1 implies a 10% random drop out of hidden units)
    "qkv_bias": False # Query-Key-Value bias (whether to include a bias vector in the Linear layers of the multi-head attention for query, key, and value computations)
}

"""
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPT2Model(
    vocab_size=GPT_CONFIG_124M['vocab_size'],
    embed_dim=GPT_CONFIG_124M['emb_dim'],
    context_length=GPT_CONFIG_124M['context_length'],
    dropout_rate=GPT_CONFIG_124M['drop_rate'],
    num_layer=GPT_CONFIG_124M['n_layers'],
    num_heads=GPT_CONFIG_124M['n_heads'],
    qkv_bias=GPT_CONFIG_124M['qkv_bias']
)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


Input batch:
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])

Output shape: torch.Size([2, 4, 50257])
tensor([[[ 0.1381,  0.0077, -0.1963,  ..., -0.0222, -0.1060,  0.1717],
         [ 0.3865, -0.8408, -0.6564,  ..., -0.5163,  0.2369, -0.3357],
         [ 0.6989, -0.1829, -0.1631,  ...,  0.1472, -0.6504, -0.0056],
         [-0.4290,  0.1669, -0.1258,  ...,  1.1579,  0.5303, -0.5549]],

        [[ 0.1094, -0.2894, -0.1467,  ..., -0.0557,  0.2911, -0.2824],
         [ 0.0882, -0.3552, -0.3527,  ...,  1.2930,  0.0053,  0.1898],
         [ 0.6091,  0.4702, -0.4094,  ...,  0.7688,  0.3787, -0.1974],
         [-0.0612, -0.0737,  0.4751,  ...,  1.2463, -0.3834,  0.0609]]],
       grad_fn=<UnsafeViewBackward0>)



total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
"""

"""Total number of parameters: 163,009,536"""

