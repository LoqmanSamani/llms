import torch
from gpt2 import GPT2
from data_loader import DataLoader



class Train(torch.nn.Module):
    def __init__(
            self,
            model,
            train_data,
            validation_data,
            batch_size,
            num_epochs,
            max_context_length,
            stride,
            shuffle,
            drop_last,
            num_workers,
            device=None,
            learning_rate=1e-3,
            weight_decay=0.1,
            vocab_text=None,
            create_vocab=False,
            encoding="gpt2",
            unk=False,
            end_of_text=False,
            vocab_start=1,
            use_custom=False,
            start_context="Education is not the learning of"
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_obj = DataLoader(
            data=train_data,
            batch_size=batch_size,
            max_length=max_context_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start,
            use_custom=use_custom
        )
        train_loader = train_obj.create_dataloader()
        validation_obj = DataLoader(
            data=validation_data,
            batch_size=batch_size,
            max_length=max_context_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            vocab_text=vocab_text,
            create_vocab=create_vocab,
            encoding=encoding,
            unk=unk,
            end_of_text=end_of_text,
            vocab_start=vocab_start,
            use_custom=use_custom
        )
        validation_loader = validation_obj.create_dataloader()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    def forward(self):
        pass

