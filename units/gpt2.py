import torch
from transformer import Transformer
from embedding import Embedding
from tokenize_ import Tokenizer
from layer_normalization import LayerNorm



class GPT2(torch.nn.Module):
    def __init__(
            self,
            input_dimension=512,
            output_dimension=512,
            num_heads=8,
            context_length=512,
            dropout_rate=0.1,
            qkv_bias=False,
            layer_norm_epsilon=1e-5,
            ff_scaling_value=4,
            num_transformers=12,
            vocab_text=None,
            create_vocab=False,
            encoding="gpt2",
            unk=False,
            end_of_text=False,
            vocab_start=1,
            vocabulary_size=512,
            embedding_dimension=512,
            use_custom=False

    ):

        super().__init__()

        self.use_custom = use_custom
        self.transformers = torch.nn.Sequential(
            *[Transformer(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                num_heads=num_heads,
                context_length=context_length,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                layer_norm_epsilon=layer_norm_epsilon,
                ff_scaling_value=ff_scaling_value
            ) for _ in range(num_transformers)]
        )
        self.tokenizer = Tokenizer(
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start
        )
        self.embedding = Embedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=embedding_dimension,
            context_length=context_length
        )
        self.final_layer_norm = LayerNorm(
            embedding_dimension=input_dimension,
            epsilon=layer_norm_epsilon
        )
        self.linear_output = torch.nn.Linear(
            in_features=input_dimension,
            out_features=vocabulary_size,
            bias=False
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, text):

        tokens = self.tokenizer.encode(
            text=text,
            use_custom=self.use_custom
        )
        tokens = self.embedding(tokens)
        tokens = self.dropout(tokens)
        tokens = self.transformers(tokens)
        tokens = self.final_layer_norm(tokens)
        logits = self.linear_output(tokens)

        return logits






