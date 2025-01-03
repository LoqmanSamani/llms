import re
import tiktoken



class Tokenizer:
    def __init__(self, vocab_text=None, create_vocab=False, encoding="gpt2", unk=False, end_of_text=False, vocab_start=1):
        """
            A tokenizer class for custom vocabularies and pre-trained `tiktoken` encodings.

            Args:
                vocab_text (str): Text to generate custom vocabulary.
                create_vocab (bool): Whether to create a custom vocabulary.
                encoding (str): Encoding name from `tiktoken`.
                unk (bool): Include `<|unk|>` token in the vocabulary.
                end_of_text (bool): Include `<|endoftext|>` token in the vocabulary.
                vocab_start (int): Starting index for vocabulary tokens.
        """

        self.encoding_names = tiktoken.list_encoding_names() # ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base']
        self.create_vocab = create_vocab
        self.encoding = encoding
        self.unk = unk
        self.end_of_text = end_of_text
        self.vocab_start = vocab_start
        if self.create_vocab and vocab_text:
            self.vocab = self.create_vocab_(
                text=vocab_text,
                unk=unk,
                end_of_text=end_of_text,
                start=vocab_start
            )


    def create_vocab_(self, text, unk, end_of_text, start):

        tokens = re.split(pattern=r'([,.:;?_!"()\']|--|\s)', string=text)
        tokens = [token.strip() for token in tokens if token.strip()]
        tokens = sorted(set(tokens))
        if unk:
            tokens.append("<|unk|>")
        if end_of_text:
            tokens.append("<|endoftext|>")

        vocab = {token: i for i, token in enumerate(tokens, start=start)}

        return vocab

    def text_to_ids(self, text, vocab):

        tokens = re.split(pattern=r'([,.:;?_!"()\']|--|\s)', string=text)
        tokens = [token.strip() for token in tokens if token.strip()]
        tokens = [token if token in vocab else "<|unk|>" for token in tokens]
        ids = [vocab[token] for token in tokens]

        return ids

    def ids_to_text(self, ids, vocab):

        reverse_vocab = {id_: token for token, id_ in vocab.items()}
        unk_id = vocab["<|unk|>"]
        tokens = ' '.join(reverse_vocab[id_] if id_ in reverse_vocab else unk_id for id_ in ids)

        return tokens


    def encode(self, text, use_custom=False):

        if use_custom:
            if not self.vocab:
                raise ValueError("Custom vocabulary is not initialized.")
            return self.text_to_ids(text=text, vocab=self.vocab)

        else:
            if self.encoding in self.encoding_names:
                tokenizer = tiktoken.get_encoding(self.encoding)
                return tokenizer.encode(text)
            raise ValueError(f"Encoding '{self.encoding}' is not valid, and no custom vocabulary is created.")

    def decode(self, ids, use_custom=False):
        if use_custom:
            if not self.vocab:
                raise ValueError("Custom vocabulary is not initialized.")
            return self.ids_to_text(ids=ids, vocab=self.vocab)

        else:
            if self.encoding in self.encoding_names:
                tokenizer = tiktoken.get_encoding(self.encoding)
                return tokenizer.decode(ids)
            raise ValueError(f"Encoding '{self.encoding}' is not valid.")





# Sample text for creating a vocabulary
vocab_text = "Hello, world! This is a tokenizer test. Can it handle unknown tokens?"
test_text = "Hello, tokenizer! Testing unknown words and end of text."

# Instantiate the Tokenizer class with a custom vocabulary
custom_tokenizer = Tokenizer(
    vocab_text=vocab_text,
    create_vocab=True,
    unk=True,  # Include <|unk|> for unknown tokens
    end_of_text=True,  # Include <|endoftext|>
    vocab_start=1  # Start indexing from 1
)




# Example 1: Custom Vocabulary Tokenization
print("Custom Vocabulary:")
print("Vocabulary:", custom_tokenizer.vocab)
"""
Vocabulary: {
    '!': 1, ',': 2, '.': 3, '?': 4, 'Can': 5, 'Hello': 6,
    'This': 7, 'a': 8, 'handle': 9, 'is': 10, 'it': 11,
    'test': 12, 'tokenizer': 13, 'tokens': 14, 'unknown': 15,
    'world': 16, '<|unk|>': 17, '<|endoftext|>': 18
    }
"""

encoded_ids_custom = custom_tokenizer.encode(test_text, use_custom=True)
print("Encoded IDs:", encoded_ids_custom)
"""Encoded IDs: [6, 2, 13, 1, 17, 15, 17, 17, 17, 17, 17, 3]"""

decoded_text_custom = custom_tokenizer.decode(encoded_ids_custom, use_custom=True)
print("Decoded Text:", decoded_text_custom)
"""Decoded Text: Hello , tokenizer ! <|unk|> unknown <|unk|> <|unk|> <|unk|> <|unk|> <|unk|> ."""


# Example 2: Pre-trained Encoding (e.g., 'gpt2')
pretrained_tokenizer = Tokenizer(encoding="gpt2")
print("\nPre-trained GPT-2 Encoding:")
encoded_ids_gpt2 = pretrained_tokenizer.encode(test_text)
print("Encoded IDs:", encoded_ids_gpt2)
"""Encoded IDs: [15496, 11, 11241, 7509, 0, 23983, 6439, 2456, 290, 886, 286, 2420, 13]"""
decoded_text_gpt2 = pretrained_tokenizer.decode(encoded_ids_gpt2)
print("Decoded Text:", decoded_text_gpt2)
"""Decoded Text: Hello, tokenizer! Testing unknown words and end of text."""






