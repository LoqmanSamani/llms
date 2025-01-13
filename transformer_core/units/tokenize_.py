import re
import tiktoken



class Tokenizer:
    def __init__(
            self,
            vocab_text=None,
            create_vocab=False,
            encoding="gpt2",
            unk=False,
            end_of_text=False,
            vocab_start=1
    ):

        self.encoding_names = tiktoken.list_encoding_names()
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
