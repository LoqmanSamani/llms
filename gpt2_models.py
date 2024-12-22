import torch
import tiktoken



"""Implements GPT-2 models"""





#                                    FORWARD PASS                                             #
###############################################################################################

class GPT2Model(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, context_length, dropout_rate, num_layer, num_heads, qkv_bias=False):
        super().__init__()

        self.token_embed = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.position_embed = torch.nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embed_dim
        )
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




#                                    TRANSFORMER BLOCK                                        #
###############################################################################################

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



#                                        LayerNorm                                            #
###############################################################################################

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




#                                        FEED FORWARD                                         #
###############################################################################################

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



#                                 GELU ACTIVATION FUNCTION                                    #
###############################################################################################

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))





#                                MULTI-HEAD ATTENTION MECHANISM                               #
###############################################################################################

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






#                                   CROSS ENTROPY LOSS                                        #
###############################################################################################

class Loss(torch.nn.Module):

    def __init__(self, model, device, temperature=0.0, top_k=None, eos_id=False):
        super().__init__()
        self.model = model
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.eos_id = eos_id # stops generating early if end-of-sequence token is encountered.

    def calc_loss_loader(self, data_loader, num_batches):

        total_loss = 0.0

        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch
                )
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches


    def calc_loss_batch(self, input_batch, target_batch):

        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)  # it calls forward function

        loss = torch.nn.functional.cross_entropy(
            input=logits.flatten(0, 1),
            target=target_batch.flatten()
        )

        return loss

    def evaluate_model(self, train_loader, val_loader, eval_iter):

        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(
                data_loader=train_loader,
                num_batches=eval_iter
            )
            val_loss = self.calc_loss_loader(
                data_loader=val_loader,
                num_batches=eval_iter
            )

        self.model.train()

        return train_loss, val_loss

    def generate(self, idx, max_new_tokens, context_size):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            if self.top_k is not None:
                top_logits, _ = torch.topk(logits, self.top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    condition=logits < min_val,
                    input=torch.tensor(float('-inf')).to(logits.device),
                    other=logits
                )

            if self.temperature > 0.0:
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == self.eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


    def text_to_token_ids(self, text, tokenizer):

        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        return encoded_tensor

    def token_ids_to_text(self, token_ids, tokenizer):

        flat = token_ids.squeeze(0)

        return tokenizer.decode(flat.tolist())


    def generate_and_print_sample(self, tokenizer, start_context):

        self.model.eval()
        context_size = self.model.position_embed.weight.shape[0]
        encoded = self.text_to_token_ids(
            text=start_context,
            tokenizer=tokenizer
        ).to(self.device)

        with torch.no_grad():
            token_ids = self.generate(
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size
            )

        decoded_text = self.token_ids_to_text(
            token_ids=token_ids,
            tokenizer=tokenizer
        )
        print(decoded_text.replace("\n", " "))
        self.model.train()






#                                        GPTDataset                                           #
###############################################################################################

class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]





#                                      GPTDataLoader                                          #
###############################################################################################

class GPTDataLoader(torch.utils.data.DataLoader):

    def __init__(self, text, batch_size, max_length, stride, shuffle, drop_last, num_workers):
        self.text = text
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    def create_dataloader(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataset(
            txt=self.text,
            tokenizer=tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

        return dataloader





#                                            TRAIN                                            #
###############################################################################################

class Train(torch.nn.Module):

    def __init__(self, model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.start_context = start_context
        self.tokenizer = tokenizer
        self.loss = Loss(
            model=self.model,
            device=self.device
        )

    def forward(self):

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(self.num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.loss.calc_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch
                )
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % self.eval_freq == 0:

                    train_loss, val_loss = self.loss.evaluate_model(
                        train_loader=self.train_loader,
                        val_loader=self.val_loader,
                        eval_iter=self.eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )

            self.loss.generate_and_print_sample(
                tokenizer=self.tokenizer,
                start_context=self.start_context
            )

        return train_losses, val_losses, track_tokens_seen







#                                    TRAIN A MODEL                                            #
###############################################################################################

"""
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = GPT2Model(
    vocab_size=GPT_CONFIG_124M['vocab_size'],
    embed_dim=GPT_CONFIG_124M['emb_dim'],
    context_length=GPT_CONFIG_124M['context_length'],
    dropout_rate=GPT_CONFIG_124M['drop_rate'],
    num_layer=GPT_CONFIG_124M['n_layers'],
    num_heads=GPT_CONFIG_124M['n_heads'],
    qkv_bias=GPT_CONFIG_124M['qkv_bias']
)

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


train_loader = GPTDataLoader(
    text=train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = GPTDataLoader(
    text=val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=0
)

train_loader = train_loader.create_dataloader()
val_loader = val_loader.create_dataloader()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)

num_epochs = 10

train = Train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    device=device,
    tokenizer=tiktoken.get_encoding("gpt2"),
    start_context="Every effort moves you"

)

train()
"""


"""
Ep 1 (Step 000000): Train loss 9.975, Val loss 9.935
Ep 1 (Step 000005): Train loss 8.073, Val loss 8.287
Every effort moves you.                                                 
Ep 2 (Step 000010): Train loss 6.657, Val loss 7.081
Ep 2 (Step 000015): Train loss 6.084, Val loss 6.621
Every effort moves you, the", the"""""", the", the, the"", the"", the, the, the"", the", the", the, the"", the"", the
Ep 3 (Step 000020): Train loss 5.568, Val loss 6.448
Ep 3 (Step 000025): Train loss 5.453, Val loss 6.359
Every effort moves you a I had to the                                             
Ep 4 (Step 000030): Train loss 5.097, Val loss 6.347
Ep 4 (Step 000035): Train loss 4.889, Val loss 6.274
Every effort moves you a, and in a little, and he was a, and in a little a little he was a--the, I had a, and he was a his a, and he was a as his a, and he was, and as his
Ep 5 (Step 000040): Train loss 4.304, Val loss 6.228
Every effort moves you know, in a little        "I turned--as he said, I felt my I felt to me, I felt him, I felt to see the picture to me to see it--as, I had been
Ep 6 (Step 000045): Train loss 3.902, Val loss 6.225
Ep 6 (Step 000050): Train loss 3.124, Val loss 6.120
Every effort moves you know, and in a        "I turned back to see it's past!                        "I I
Ep 7 (Step 000055): Train loss 2.736, Val loss 6.175
Ep 7 (Step 000060): Train loss 2.412, Val loss 6.214
Every effort moves you know you know you know the picture for a smile that he was the last--as I had been his pictures--I didn't you know, I had been his glory, I had been the honour of the donkey, and I had a small picture
Ep 8 (Step 000065): Train loss 2.159, Val loss 6.242
Ep 8 (Step 000070): Train loss 1.702, Val loss 6.256
Every effort moves you?"  "--as he was a smile that he was.  "I turned work, and in the fact the picture to the picture.                "I didn
Ep 9 (Step 000075): Train loss 1.325, Val loss 6.295
Ep 9 (Step 000080): Train loss 1.129, Val loss 6.319
Every effort moves you?"  "Yes--as I felt to the fact with such--had not existed till nearly a year after Jack's the cigars you like."                 "I found
Ep 10 (Step 000085): Train loss 0.750, Val loss 6.314
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him. "Oh, and went on groping and muddling; then I looked at the donkey again. I saw that, my eye fell on a small picture

"""

"""
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
"""

"""Total number of parameters: 163,009,536"""
"""
for name, param in model.state_dict().items():
    print(name)
"""


"""
token_embed.weight
position_embed.weight
transformer_blocks.0.attention.mask
transformer_blocks.0.attention.W_query.weight
transformer_blocks.0.attention.W_key.weight
transformer_blocks.0.attention.W_value.weight
transformer_blocks.0.attention.out_projection.weight
transformer_blocks.0.attention.out_projection.bias
transformer_blocks.0.feed_forward.layers.0.weight
transformer_blocks.0.feed_forward.layers.0.bias
transformer_blocks.0.feed_forward.layers.2.weight
transformer_blocks.0.feed_forward.layers.2.bias
transformer_blocks.0.norm1.scale
transformer_blocks.0.norm1.shift
transformer_blocks.0.norm2.scale
transformer_blocks.0.norm2.shift
transformer_blocks.1.attention.mask
transformer_blocks.1.attention.W_query.weight
transformer_blocks.1.attention.W_key.weight
transformer_blocks.1.attention.W_value.weight
transformer_blocks.1.attention.out_projection.weight
transformer_blocks.1.attention.out_projection.bias
transformer_blocks.1.feed_forward.layers.0.weight
transformer_blocks.1.feed_forward.layers.0.bias
transformer_blocks.1.feed_forward.layers.2.weight
transformer_blocks.1.feed_forward.layers.2.bias
transformer_blocks.1.norm1.scale
transformer_blocks.1.norm1.shift
transformer_blocks.1.norm2.scale
transformer_blocks.1.norm2.shift
transformer_blocks.2.attention.mask
transformer_blocks.2.attention.W_query.weight
transformer_blocks.2.attention.W_key.weight
transformer_blocks.2.attention.W_value.weight
transformer_blocks.2.attention.out_projection.weight
transformer_blocks.2.attention.out_projection.bias
transformer_blocks.2.feed_forward.layers.0.weight
transformer_blocks.2.feed_forward.layers.0.bias
transformer_blocks.2.feed_forward.layers.2.weight
transformer_blocks.2.feed_forward.layers.2.bias
transformer_blocks.2.norm1.scale
transformer_blocks.2.norm1.shift
transformer_blocks.2.norm2.scale
transformer_blocks.2.norm2.shift
transformer_blocks.3.attention.mask
transformer_blocks.3.attention.W_query.weight
transformer_blocks.3.attention.W_key.weight
transformer_blocks.3.attention.W_value.weight
transformer_blocks.3.attention.out_projection.weight
transformer_blocks.3.attention.out_projection.bias
transformer_blocks.3.feed_forward.layers.0.weight
transformer_blocks.3.feed_forward.layers.0.bias
transformer_blocks.3.feed_forward.layers.2.weight
transformer_blocks.3.feed_forward.layers.2.bias
transformer_blocks.3.norm1.scale
transformer_blocks.3.norm1.shift
transformer_blocks.3.norm2.scale
transformer_blocks.3.norm2.shift
transformer_blocks.4.attention.mask
transformer_blocks.4.attention.W_query.weight
transformer_blocks.4.attention.W_key.weight
transformer_blocks.4.attention.W_value.weight
transformer_blocks.4.attention.out_projection.weight
transformer_blocks.4.attention.out_projection.bias
transformer_blocks.4.feed_forward.layers.0.weight
transformer_blocks.4.feed_forward.layers.0.bias
transformer_blocks.4.feed_forward.layers.2.weight
transformer_blocks.4.feed_forward.layers.2.bias
transformer_blocks.4.norm1.scale
transformer_blocks.4.norm1.shift
transformer_blocks.4.norm2.scale
transformer_blocks.4.norm2.shift
transformer_blocks.5.attention.mask
transformer_blocks.5.attention.W_query.weight
transformer_blocks.5.attention.W_key.weight
transformer_blocks.5.attention.W_value.weight
transformer_blocks.5.attention.out_projection.weight
transformer_blocks.5.attention.out_projection.bias
transformer_blocks.5.feed_forward.layers.0.weight
transformer_blocks.5.feed_forward.layers.0.bias
transformer_blocks.5.feed_forward.layers.2.weight
transformer_blocks.5.feed_forward.layers.2.bias
transformer_blocks.5.norm1.scale
transformer_blocks.5.norm1.shift
transformer_blocks.5.norm2.scale
transformer_blocks.5.norm2.shift
transformer_blocks.6.attention.mask
transformer_blocks.6.attention.W_query.weight
transformer_blocks.6.attention.W_key.weight
transformer_blocks.6.attention.W_value.weight
transformer_blocks.6.attention.out_projection.weight
transformer_blocks.6.attention.out_projection.bias
transformer_blocks.6.feed_forward.layers.0.weight
transformer_blocks.6.feed_forward.layers.0.bias
transformer_blocks.6.feed_forward.layers.2.weight
transformer_blocks.6.feed_forward.layers.2.bias
transformer_blocks.6.norm1.scale
transformer_blocks.6.norm1.shift
transformer_blocks.6.norm2.scale
transformer_blocks.6.norm2.shift
transformer_blocks.7.attention.mask
transformer_blocks.7.attention.W_query.weight
transformer_blocks.7.attention.W_key.weight
transformer_blocks.7.attention.W_value.weight
transformer_blocks.7.attention.out_projection.weight
transformer_blocks.7.attention.out_projection.bias
transformer_blocks.7.feed_forward.layers.0.weight
transformer_blocks.7.feed_forward.layers.0.bias
transformer_blocks.7.feed_forward.layers.2.weight
transformer_blocks.7.feed_forward.layers.2.bias
transformer_blocks.7.norm1.scale
transformer_blocks.7.norm1.shift
transformer_blocks.7.norm2.scale
transformer_blocks.7.norm2.shift
transformer_blocks.8.attention.mask
transformer_blocks.8.attention.W_query.weight
transformer_blocks.8.attention.W_key.weight
transformer_blocks.8.attention.W_value.weight
transformer_blocks.8.attention.out_projection.weight
transformer_blocks.8.attention.out_projection.bias
transformer_blocks.8.feed_forward.layers.0.weight
transformer_blocks.8.feed_forward.layers.0.bias
transformer_blocks.8.feed_forward.layers.2.weight
transformer_blocks.8.feed_forward.layers.2.bias
transformer_blocks.8.norm1.scale
transformer_blocks.8.norm1.shift
transformer_blocks.8.norm2.scale
transformer_blocks.8.norm2.shift
transformer_blocks.9.attention.mask
transformer_blocks.9.attention.W_query.weight
transformer_blocks.9.attention.W_key.weight
transformer_blocks.9.attention.W_value.weight
transformer_blocks.9.attention.out_projection.weight
transformer_blocks.9.attention.out_projection.bias
transformer_blocks.9.feed_forward.layers.0.weight
transformer_blocks.9.feed_forward.layers.0.bias
transformer_blocks.9.feed_forward.layers.2.weight
transformer_blocks.9.feed_forward.layers.2.bias
transformer_blocks.9.norm1.scale
transformer_blocks.9.norm1.shift
transformer_blocks.9.norm2.scale
transformer_blocks.9.norm2.shift
transformer_blocks.10.attention.mask
transformer_blocks.10.attention.W_query.weight
transformer_blocks.10.attention.W_key.weight
transformer_blocks.10.attention.W_value.weight
transformer_blocks.10.attention.out_projection.weight
transformer_blocks.10.attention.out_projection.bias
transformer_blocks.10.feed_forward.layers.0.weight
transformer_blocks.10.feed_forward.layers.0.bias
transformer_blocks.10.feed_forward.layers.2.weight
transformer_blocks.10.feed_forward.layers.2.bias
transformer_blocks.10.norm1.scale
transformer_blocks.10.norm1.shift
transformer_blocks.10.norm2.scale
transformer_blocks.10.norm2.shift
transformer_blocks.11.attention.mask
transformer_blocks.11.attention.W_query.weight
transformer_blocks.11.attention.W_key.weight
transformer_blocks.11.attention.W_value.weight
transformer_blocks.11.attention.out_projection.weight
transformer_blocks.11.attention.out_projection.bias
transformer_blocks.11.feed_forward.layers.0.weight
transformer_blocks.11.feed_forward.layers.0.bias
transformer_blocks.11.feed_forward.layers.2.weight
transformer_blocks.11.feed_forward.layers.2.bias
transformer_blocks.11.norm1.scale
transformer_blocks.11.norm1.shift
transformer_blocks.11.norm2.scale
transformer_blocks.11.norm2.shift
final_norm.scale
final_norm.shift
linear_transform.weight
"""